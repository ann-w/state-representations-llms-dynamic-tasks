from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Tuple
import numpy as np
from PIL import Image, ImageDraw
import messenger

# Reconstructed clean renderer with compact HUD.

def _load_asset(assets_dir: Path, name: str, required: bool = False) -> Optional[Image.Image]:
    for ext in ("jpeg", "jpg", "png"):
        p = assets_dir / f"{name}.{ext}"
        if p.exists():
            try: return Image.open(p).convert("RGBA")
            except Exception: continue
    if required: raise FileNotFoundError(f"Asset missing: {name}")
    return None

def render_image_grid(env, obs, image_language_mode: bool = False, transitional_prev: Optional[Tuple[int,int]] = None, transitional_stage: bool = False) -> str:
    from messenger.envs import config as ms_config
    assets_dir = Path(messenger.__file__).parent / "images"
    if not assets_dir.exists(): raise FileNotFoundError(str(assets_dir))
    load = lambda n: _load_asset(assets_dir, n)
    # Tiles
    tile_grass = load("grass"); tile_wall = load("wall")
    base_bg_tile = tile_grass or tile_wall or Image.new("RGBA", (32,32),(34,68,34,255))
    tile_agent = load("agent"); tile_agent_up = load("agent_up"); tile_agent_down = load("agent_down")
    tile_agent_left = load("agent_left"); tile_agent_right = load("agent_right"); tile_agent_msg = load("agent_with_message")
    tile_message = load("message"); tile_goal = load("goal"); tile_neutral = load("npc"); tile_npc = tile_neutral or base_bg_tile
    def ensure_placeholder(img, color, label):
        if img is not None: return img
        w,h = base_bg_tile.size; ph = Image.new("RGBA", (w,h), color)
        try: ImageDraw.Draw(ph).text((4,4), label, fill=(255,255,255,255))
        except Exception: pass
        return ph
    tile_agent = ensure_placeholder(tile_agent,(40,40,220,255),"A")
    tile_agent_msg = ensure_placeholder(tile_agent_msg,(220,180,30,255),"AM")
    tile_goal = ensure_placeholder(tile_goal,(40,200,40,255),"G")
    def maybe_resize(im):
        if im is None: return im
        if env._force_tile_size and isinstance(env._force_tile_size,int):
            t=int(env._force_tile_size)
            if im.width>t or im.height>t: im = im.resize((t,t), Image.BICUBIC)
        return im
    for name in ["base_bg_tile","tile_agent","tile_agent_up","tile_agent_down","tile_agent_left","tile_agent_right","tile_agent_msg","tile_message","tile_goal","tile_npc","tile_wall"]:
        if name in locals(): locals()[name]=maybe_resize(locals()[name])
    base_bg_tile=locals()["base_bg_tile"]; tile_wall=locals().get("tile_wall"); tile_message=locals()["tile_message"]; tile_goal=locals()["tile_goal"]; tile_npc=locals()["tile_npc"]
    tw,th = base_bg_tile.size
    if (tw>48 or th>48) and not env._force_tile_size:
        target=48; scale=min(target/tw,target/th)
        if scale<1:
            ntw,nth=max(8,int(tw*scale)),max(8,int(th*scale))
            def rs(im): return im.resize((ntw,nth),Image.BICUBIC) if im and (im.width!=ntw or im.height!=nth) else im
            for name in ["base_bg_tile","tile_agent","tile_agent_up","tile_agent_down","tile_agent_left","tile_agent_right","tile_agent_msg","tile_message","tile_goal","tile_npc","tile_wall"]:
                im=locals().get(name); locals()[name]=rs(im) if im is not None else im
            base_bg_tile=locals()["base_bg_tile"]; tw,th=base_bg_tile.size
    entities_map=obs["entities"]; entities_map_2d=entities_map.max(axis=2) if entities_map.ndim==3 else entities_map
    H,W=entities_map_2d.shape[:2]; canvas=Image.new("RGBA",(W*tw,H*th))
    a15=np.where(obs["avatar"]==15); a16=np.where(obs["avatar"]==16)
    if len(a15[0])>0: ay,ax=int(a15[0][0]),int(a15[1][0]); agent_has_message=False
    elif len(a16[0])>0: ay,ax=int(a16[0][0]),int(a16[1][0]); agent_has_message=True
    else: ay,ax=-1,-1; agent_has_message=False
    game=getattr(env._env,"game",None)
    enemy_id=getattr(getattr(game,"enemy",None),"id",None) if game else None
    message_id=getattr(getattr(game,"message",None),"id",None) if game else None
    goal_id=getattr(getattr(game,"goal",None),"id",None) if game else None
    from messenger.envs import config as _cfg
    id_to_name={e.id:e.name for e in getattr(_cfg,"NPCS",[])}
    if not hasattr(env,"_missing_variants") or env._missing_variants is None:
        try: env._missing_variants=set()
        except Exception: pass
    sprite_cache: Dict[str,Optional[Image.Image]]={}
    def load_cached(base:str):
        if base in sprite_cache: return sprite_cache[base]
        img=load(base); sprite_cache[base]=img; return img
    show_enemy_variant=bool(getattr(env,"_enemy_collision_reveal",False) or getattr(env,"_just_done",False))
    for y in range(H):
        for x in range(W):
            px,py=x*tw,y*th; canvas.alpha_composite(base_bg_tile,(px,py)); eid=int(entities_map_2d[y,x])
            if eid==0: continue
            if tile_wall is not None and eid==ms_config.WALL.id: canvas.alpha_composite(tile_wall,(px,py)); continue
            if eid in (15,16): continue
            ent_name=id_to_name.get(eid); applied=False
            if image_language_mode and ent_name and message_id is not None and eid==message_id and not agent_has_message: ent_name=None
            if ent_name:
                is_enemy=(enemy_id is not None and eid==enemy_id); is_goal=(goal_id is not None and eid==goal_id); is_msg=(message_id is not None and eid==message_id)
                friendly=f"{ent_name}_friendly"; enemy=f"{ent_name}_enemy"; chosen=None
                if is_enemy: chosen=(load_cached(enemy) if show_enemy_variant else None) or load_cached(friendly)
                elif is_goal: chosen=load_cached(friendly)
                elif is_msg: chosen=load_cached(friendly)
                else: chosen=load_cached(friendly)
                if chosen is not None: canvas.alpha_composite(chosen,(px,py)); applied=True
                else:
                    if getattr(env,'debug_assets',False):
                        if friendly not in env._missing_variants:
                            try: env._missing_variants.add(friendly)
                            except Exception: pass
                            print(f"[ImageRenderer][MISSING VARIANT] {friendly}.* not found")
                        if is_enemy and show_enemy_variant and enemy not in env._missing_variants:
                            try: env._missing_variants.add(enemy)
                            except Exception: pass
                            print(f"[ImageRenderer][MISSING VARIANT] {enemy}.* not found")
            if not applied:
                if goal_id is not None and eid==goal_id and tile_goal is not None: canvas.alpha_composite(tile_goal,(px,py)); applied=True
                elif message_id is not None and eid==message_id and tile_message is not None: canvas.alpha_composite(tile_message,(px,py)); applied=True
        try:
            py_prev,px_prev=transitional_prev
            if 0<=py_prev<H and 0<=px_prev<W and (py_prev,px_prev)!=(ay,ax):
                gx,gy=px_prev*tw,py_prev*th; ghost=(tile_agent_msg if agent_has_message else tile_agent).copy()
                alpha=ghost.split()[-1].point(lambda a:int(a*0.35)); ghost.putalpha(alpha); canvas.alpha_composite(ghost,(gx,gy))
        except Exception: pass
    if 0<=ay<H and 0<=ax<W:
        px,py=ax*tw,ay*th; agent_base=tile_agent
        if env._last_action is not None:
            try:
                an=env.action_list[env._last_action]
                if an=="Move North" and tile_agent_up is not None: agent_base=tile_agent_up
                elif an=="Move South" and tile_agent_down is not None: agent_base=tile_agent_down
                elif an=="Move West" and tile_agent_left is not None: agent_base=tile_agent_left
                elif an=="Move East" and tile_agent_right is not None: agent_base=tile_agent_right
            except Exception: pass
        # If episode ended successfully, show goal sprite instead of agent to emphasize success
        success = getattr(env,'_last_termination_reason',None) == 'reach_goal_with_message'
        if success and tile_goal is not None:
            try:
                canvas.alpha_composite(tile_goal,(px,py))
            except Exception:
                canvas.alpha_composite(agent_base,(px,py))
        else:
            if agent_has_message and tile_agent_msg is not None:
                try: canvas.alpha_composite(agent_base,(px,py)); canvas.alpha_composite(tile_agent_msg,(px,py))
                except Exception: canvas.alpha_composite(agent_base,(px,py))
            else: canvas.alpha_composite(agent_base,(px,py))
        # Persistent message icon: show on original entity tile (not moving with agent) if origin known
        if agent_has_message and env._show_message_persistent and tile_message is not None:
            origin = getattr(env, '_message_origin_pos', None)
            if origin is not None:
                oy, ox = origin
                if 0 <= oy < H and 0 <= ox < W:
                    try: canvas.alpha_composite(tile_message, (ox*tw, oy*th))
                    except Exception: pass
    if getattr(env,'enable_flashes',False) and getattr(env,'_flash_queue',None):
        try:
            flash_img=env._flash_queue.pop(0)
            # Use pickup target position for pickup flashes, otherwise agent position
            target = getattr(env, '_flash_pickup_target', None) if getattr(env, '_flash_queue_type', None) == 'pickup' else ((ay, ax) if (0<=ay<H and 0<=ax<W) else None)
            if target and flash_img.size==(tw,th):
                ty, tx = target
                if 0<=ty<H and 0<=tx<W:
                    canvas.alpha_composite(flash_img,(tx*tw,ty*th))
            else:
                fin=flash_img if flash_img.size==canvas.size else flash_img.resize(canvas.size,Image.NEAREST)
                canvas.alpha_composite(fin,(0,0))
        except Exception: pass
    try:
        total_px=canvas.width*canvas.height
        if env.max_render_pixels and total_px>int(env.max_render_pixels):
            import math
            scale=math.sqrt(env.max_render_pixels/total_px); nw=max(1,int(canvas.width*scale)); nh=max(1,int(canvas.height*scale))
            if nw<canvas.width and nh<canvas.height: canvas=canvas.resize((nw,nh),Image.BICUBIC)
    except Exception: pass
    # HUD
    try:
        from PIL import ImageFont
        hud_h=max(4,int(th*0.75))
        final_canvas=Image.new("RGBA",(canvas.width,canvas.height+hud_h),(255,255,255,0))
        final_canvas.paste(canvas,(0,hud_h))
        draw_hud=ImageDraw.Draw(final_canvas)
        draw_hud.rectangle([0,0,final_canvas.width,hud_h],fill=(255,255,255,245))
        reason=getattr(env,'_last_termination_reason',None)
        steps=getattr(env,'_action_steps',0)
        base_scale=0.55*0.75*0.5
        font_px=max(6,int(th*base_scale))
        def load_font(sz):
            for name in ("DejaVuSans.ttf","Arial.ttf"):
                try: return ImageFont.truetype(name,sz)
                except Exception: continue
            return ImageFont.load_default()
        font=load_font(font_px)
        left_text=f"Step:{steps} | Score:{getattr(env,'_current_score',0.0):.2f}"
        right_text=f"Status:{reason}" if reason else ""
        # Horizontal padding inside HUD for left and right text blocks
        margin=12; gap_min=10
        def measure(f):
            l_box=draw_hud.textbbox((0,0),left_text,font=f); r_box=draw_hud.textbbox((0,0),right_text,font=f) if right_text else None; return l_box,r_box
        def total_w(l_b,r_b):
            lw=l_b[2]
            if not right_text: return lw+margin*2
            rw=r_b[2]; return lw+gap_min+rw+margin*2
        l_box,r_box=measure(font)
        while total_w(l_box,r_box)>final_canvas.width and font_px>6:
            new_px=max(6,int(font_px*0.9))
            if new_px==font_px: break
            font_px=new_px; font=load_font(font_px); l_box,r_box=measure(font)
        if right_text and total_w(l_box,r_box)>final_canvas.width:
            ellipsis='…'; txt=right_text
            while len(txt)>5 and total_w(l_box,r_box)>final_canvas.width:
                txt=txt[:-4]+ellipsis; right_text=txt; l_box,r_box=measure(font)
        l_h=l_box[3]-l_box[1]; r_h=(r_box[3]-r_box[1]) if (right_text and r_box) else 0
        text_h=max(l_h,r_h); text_y=max(0,(hud_h-text_h)//2)
        draw_hud.text((margin,text_y),left_text,fill=(0,0,0,255),font=font)
        if right_text and r_box:
            rw=r_box[2]; rx=final_canvas.width-margin-rw; draw_hud.text((rx,text_y),right_text,fill=(0,0,0,255),font=font)
        canvas=final_canvas
    except Exception: pass
    out_dir = env._frame_output_dir if getattr(env,"_frame_output_dir",None) is not None else (assets_dir/"renders")
    try: out_dir.mkdir(parents=True,exist_ok=True)
    except Exception: pass
    frame_idx=getattr(env,"_frame_index",0); out_path=out_dir/f"frame_{frame_idx:05d}.jpeg"
    try: env._frame_index=frame_idx+1
    except Exception: pass
    canvas.convert("RGB").save(out_path,quality=95,optimize=True)
    return str(out_path.resolve())

from PIL import Image  # flash helpers
def _flash_assets_dir()->Path: return Path(messenger.__file__).parent/"images"
def load_flash_sequence(env,flash_type:str):
    if flash_type not in ("pickup","fail","success"): return []
    attr=f"_flash_cache_{flash_type}";
    if hasattr(env,attr):
        cached=getattr(env,attr)
        if cached is not None: return cached
    assets=_flash_assets_dir(); prefix=f"{flash_type}_flash_"; frames=[]
    if assets.exists():
        candidates=list(assets.glob(f"{prefix}*.jpeg"))
        def sk(p):
            try: return int(p.stem.replace(prefix,""))
            except Exception: return 0
        for p in sorted(candidates,key=sk):
            try: frames.append(Image.open(p).convert("RGBA"))
            except Exception: continue
    setattr(env,attr,frames); return frames
def queue_flash_sequence(env,flash_type:str):
    seq=load_flash_sequence(env,flash_type)
    if seq:
        if not hasattr(env,"_flash_queue") or env._flash_queue is None: env._flash_queue=[]
        env._flash_queue=[im.copy() for im in seq]
def queue_message_pickup_flash(env):
    try:
        assets=_flash_assets_dir(); frame_paths=[assets/"message_pickup_flash_0.jpeg", assets/"message_pickup_flash_1.jpeg"]
        frames=[]
        for p in frame_paths:
            if p.exists():
                try: frames.append(Image.open(p).convert("RGBA"))
                except Exception: pass
        if not frames:
            frames=load_flash_sequence(env,"pickup")[:2]
        if frames:
            if not hasattr(env,"_flash_queue") or env._flash_queue is None: env._flash_queue=[]
            env._flash_queue.extend([im.copy() for im in frames[:2]])
    except Exception: pass

