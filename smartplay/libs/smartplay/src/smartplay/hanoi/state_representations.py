from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Any, Optional, Union
import os
import json
import re


class StateRepresentation(ABC):
    """
    Abstract base class for different Hanoi state representations.
    """

    @abstractmethod
    def from_internal_state(
        self, internal_state: Tuple[int, ...], num_disks: int
    ) -> Any:
        """
        Converts the environment's internal state (tuple of peg_ids for each disk)
        to this specific representation.
        Example internal_state for 3 disks: (0, 1, 2) means disk 0 on peg A, disk 1 on peg B, disk 2 on peg C.
        Disks are numbered 0 (smallest) to num_disks-1 (largest).
        Pegs are 0 (A), 1 (B), 2 (C).
        """
        pass

    @abstractmethod
    def to_internal_state(self, custom_state: Any, num_disks: int) -> Tuple[int, ...]:
        """
        Converts this specific representation back to the environment's internal state.
        """
        pass

    @abstractmethod
    def describe(self, custom_state: Any) -> str:
        """
        Returns a human-readable string description of the state in this representation.
        """
        pass


class DefaultStateRepresentation(StateRepresentation):
    """Default rod-based representation: - A: |bottom, [disks], top|"""

    def from_internal_state(self, internal_state, num_disks):
        # Create lists for each peg
        pegs = [[], [], []]

        # Distribute disks to pegs (largest disk ID = largest disk)
        for disk_id in range(num_disks):
            peg = internal_state[disk_id]
            pegs[peg].append(disk_id)

        # Sort each peg so largest disks are at bottom (reverse order)
        for peg in pegs:
            peg.sort(reverse=True)

        # Return as rod format string, not dictionary
        return f"- A: |bottom, {pegs[0]}, top|\n- B: |bottom, {pegs[1]}, top|\n- C: |bottom, {pegs[2]}, top|"

    def to_internal_state(self, external_state, num_disks):
        # Parse the rod format string back to internal state
        lines = external_state.strip().split("\n")
        pegs = [[], [], []]

        for i, line in enumerate(lines):
            # Extract the list part between brackets
            start = line.find("[")
            end = line.find("]")
            if start != -1 and end != -1:
                list_str = line[start + 1 : end].strip()
                if list_str:
                    # Parse the comma-separated disk IDs
                    disk_ids = [
                        int(x.strip()) for x in list_str.split(",") if x.strip()
                    ]
                    pegs[i] = disk_ids

        # Convert back to internal state tuple
        internal_state = [0] * num_disks
        for peg_id, disk_list in enumerate(pegs):
            for disk_id in disk_list:
                internal_state[disk_id] = peg_id

        return tuple(internal_state)

    def describe(self, external_state):
        # external_state is already the formatted string
        return external_state


class DictListStateRepresentation(StateRepresentation):
    """
    Dictionary of lists representation.
    Example: {"A": [], "B": [1], "C": [2, 0]}
    Lists show disks from bottom to top (largest to smallest).
    """

    PEG_NAMES = ["A", "B", "C"]

    def from_internal_state(
        self, internal_state: Tuple[int, ...], num_disks: int
    ) -> Dict[str, List[int]]:
        pegs: Dict[str, List[int]] = {name: [] for name in self.PEG_NAMES}

        # Group disks by peg
        for disk_id, peg_idx in enumerate(internal_state):
            pegs[self.PEG_NAMES[peg_idx]].append(disk_id)

        # Sort disks on each peg (largest to smallest, bottom to top)
        for peg_name in pegs:
            pegs[peg_name].sort(reverse=True)

        return pegs

    def to_internal_state(
        self, custom_state: Dict[str, List[int]], num_disks: int
    ) -> Tuple[int, ...]:
        internal_state_list = [-1] * num_disks
        peg_map = {name: i for i, name in enumerate(self.PEG_NAMES)}

        for peg_name, disks_on_peg in custom_state.items():
            peg_idx = peg_map[peg_name]
            for disk_id in disks_on_peg:
                if 0 <= disk_id < num_disks:
                    internal_state_list[disk_id] = peg_idx
                else:
                    raise ValueError(
                        f"Invalid disk_id {disk_id} found in custom state."
                    )

        if any(p == -1 for p in internal_state_list):
            raise ValueError(
                "Not all disks were assigned to a peg in the custom state."
            )

        return tuple(internal_state_list)

    def describe(self, custom_state: Dict[str, List[int]]) -> str:
        return str(custom_state)


class MatrixStateRepresentation(StateRepresentation):
    """
    Matrix representation where each row is a peg and each column is a disk position.
    Example: [[-1,-1,-1], [-1,-1, 1], [ 2,-1, 0]]
    -1 means empty slot, numbers are disk IDs.
    Left to right is bottom to top.
    """

    def from_internal_state(
        self, internal_state: Tuple[int, ...], num_disks: int
    ) -> List[List[int]]:
        # Initialize matrix: 3 pegs, each can hold up to num_disks
        matrix_state: List[List[int]] = [[-1] * num_disks for _ in range(3)]

        # Group disks by peg
        temp_pegs: List[List[int]] = [[] for _ in range(3)]
        for disk_id, peg_idx in enumerate(internal_state):
            temp_pegs[peg_idx].append(disk_id)

        # Fill matrix: largest disks at leftmost positions (bottom)
        for peg_idx, disks_on_peg in enumerate(temp_pegs):
            sorted_disks = sorted(disks_on_peg, reverse=True)  # largest to smallest
            for i, disk_id in enumerate(sorted_disks):
                matrix_state[peg_idx][i] = disk_id

        return matrix_state

    def to_internal_state(
        self, custom_state: List[List[int]], num_disks: int
    ) -> Tuple[int, ...]:
        if len(custom_state) != 3:
            raise ValueError(f"Matrix state must have 3 rows (pegs).")
        if any(len(peg_row) != num_disks for peg_row in custom_state):
            raise ValueError(
                f"Each row in matrix state must have {num_disks} columns (disk slots)."
            )

        internal_state_list = [-1] * num_disks

        for peg_idx, peg_row in enumerate(custom_state):
            for disk_id in peg_row:
                if disk_id != -1:  # -1 means empty slot
                    if not (0 <= disk_id < num_disks):
                        raise ValueError(f"Invalid disk_id {disk_id} in matrix.")
                    if internal_state_list[disk_id] != -1:
                        raise ValueError(f"Disk {disk_id} found on multiple pegs.")
                    internal_state_list[disk_id] = peg_idx

        if any(p == -1 for p in internal_state_list):
            unplaced_disks = [
                i for i, p_idx in enumerate(internal_state_list) if p_idx == -1
            ]
            raise ValueError(f"Disks {unplaced_disks} not found in matrix state.")

        return tuple(internal_state_list)

    def describe(self, custom_state: List[List[int]]) -> str:
        return str(custom_state)


class NaturalLanguageStateRepresentation(StateRepresentation):
    """
    Free-text representation, e.g.:
       "Peg A is empty. Peg B has disk 1 on top. Peg C has disk 2 at the bottom and disk 0 on top."
    • Order in every sentence is bottom → … → top.
    • Disk IDs must appear as integers.
    """

    def from_internal_state(
        self, internal_state: Tuple[int, ...], num_disks: int
    ) -> str:
        # Create lists for each peg
        pegs = [[], [], []]

        # Distribute disks to pegs (largest disk ID = largest disk)
        for disk_id in range(num_disks):
            peg = internal_state[disk_id]
            pegs[peg].append(disk_id)

        # Sort each peg so largest disks are at bottom (reverse order)
        for peg in pegs:
            peg.sort(reverse=True)

        # Generate natural language description
        peg_names = ["A", "B", "C"]
        sentences = []

        for i, (peg_name, disks) in enumerate(zip(peg_names, pegs)):
            if not disks:
                sentences.append(f"Peg {peg_name} is empty.")
            elif len(disks) == 1:
                sentences.append(f"Peg {peg_name} has disk {disks[0]} on top.")
            else:
                # Multiple disks: describe from bottom to top
                if len(disks) == 2:
                    sentences.append(
                        f"Peg {peg_name} has disk {disks[0]} at the bottom and disk {disks[1]} on top."
                    )
                else:
                    # For 3+ disks: "disk X at the bottom, disk Y in the middle, and disk Z on top"
                    bottom_disk = disks[0]
                    top_disk = disks[-1]
                    middle_disks = disks[1:-1]

                    if len(middle_disks) == 1:
                        sentences.append(
                            f"Peg {peg_name} has disk {bottom_disk} at the bottom, disk {middle_disks[0]} in the middle, and disk {top_disk} on top."
                        )
                    else:
                        middle_str = ", ".join([f"disk {d}" for d in middle_disks])
                        sentences.append(
                            f"Peg {peg_name} has disk {bottom_disk} at the bottom, {middle_str}, and disk {top_disk} on top."
                        )

        return " ".join(sentences)

    def to_internal_state(self, external_state: str, num_disks: int) -> Tuple[int, ...]:
        import re

        # Initialize internal state
        internal_state = [0] * num_disks

        # Parse each peg's description
        peg_names = ["A", "B", "C"]

        for peg_idx, peg_name in enumerate(peg_names):
            # Find the sentence about this peg
            peg_pattern = rf"Peg {peg_name} (.*?)\."
            match = re.search(peg_pattern, external_state)

            if match:
                peg_description = match.group(1)

                if "is empty" in peg_description:
                    continue  # No disks on this peg

                # Extract all disk numbers from the description
                disk_pattern = r"disk (\d+)"
                disk_matches = re.findall(disk_pattern, peg_description)
                disk_ids = [int(d) for d in disk_matches]

                # Assign these disks to this peg
                for disk_id in disk_ids:
                    if 0 <= disk_id < num_disks:
                        internal_state[disk_id] = peg_idx
                    else:
                        raise ValueError(
                            f"Invalid disk_id {disk_id} found in natural language state."
                        )

        return tuple(internal_state)

    def describe(self, external_state: str) -> str:
        # external_state is already the formatted string
        return external_state


class LuaFunctionStateRepresentation(StateRepresentation):
    """
    Represent the Hanoi state as a Lua function that returns a table
    with three list fields: A, B, C (bottom → top order).
    """

    def from_internal_state(
        self, internal_state: Tuple[int, ...], num_disks: int
    ) -> str:
        # 1. gather disks on each peg
        pegs = {0: [], 1: [], 2: []}
        for disk in range(num_disks):
            pegs[internal_state[disk]].append(disk)

        # 2. bottom-to-top ordering (largest-ID at index 1)
        #    pegs[i] is already bottom->top because disks are numbered
        #    small ID = small disk; if you invert that rule, sort reverse.
        def to_lua_list(lst):
            return "{" + ", ".join(map(str, lst[::-1])) + "}"  # top → right

        lua_table = (
            "function get_state()\n"
            "  return {\n"
            f"    A = {to_lua_list(pegs[0])},\n"
            f"    B = {to_lua_list(pegs[1])},\n"
            f"    C = {to_lua_list(pegs[2])}\n"
            "  }\n"
            "end"
        )
        return lua_table

    def to_internal_state(self, custom_state: str, num_disks: int) -> Tuple[int, ...]:
        """
        Parse a Lua function representation back to internal state.
        """
        import re

        # Extract the lists for each peg from the Lua function
        peg_pattern = r"([ABC])\s*=\s*{([^}]*)}"
        matches = re.findall(peg_pattern, custom_state)

        # Create a mapping of peg name to list of disks
        peg_map = {'A': 0, 'B': 1, 'C': 2}
        pegs = {0: [], 1: [], 2: []}

        for peg_name, disk_list in matches:
            peg_idx = peg_map[peg_name]
            # Parse the disk list (may be empty)
            if disk_list.strip():
                # Split and convert to integers, reversing to get bottom-to-top order
                disks = [int(d.strip()) for d in disk_list.split(',')]
                pegs[peg_idx] = disks[::-1]  # Reverse to get bottom→top

        # Initialize the internal state
        internal_state = [-1] * num_disks

        # Assign each disk to its peg
        for peg_idx, disks in pegs.items():
            for disk_id in disks:
                if 0 <= disk_id < num_disks:
                    internal_state[disk_id] = peg_idx
                else:
                    raise ValueError(f"Invalid disk_id {disk_id} in Lua representation")

        # Ensure all disks are assigned
        if -1 in internal_state:
            missing_disks = [i for i, p in enumerate(internal_state) if p == -1]
            raise ValueError(f"Disks {missing_disks} not found in Lua representation")

        return tuple(internal_state)

    def describe(self, formatted_state: str) -> str:
        return formatted_state


# New: Image-based state representation using pre-rendered sprites
class ImageStateRepresentation(StateRepresentation):
    """
    Represents the Hanoi state as a reference to a pre-rendered image on disk.

    - from_internal_state returns a JSON-serializable dict with id, vector, filename, and absolute path.
    - to_internal_state accepts either that dict or a filename/id and converts back to the tuple state.

    Limitations:
    • The bundled image set currently supports exactly 3 disks and 3 pegs.
    """

    def __init__(self, images_dir: Optional[str] = None, metadata_filename: str = "metadata.json"):
        base_dir = os.path.dirname(__file__)
        self.images_dir = images_dir or os.path.join(base_dir, "images")
        self.metadata_path = os.path.join(self.images_dir, metadata_filename)
        self._meta = None
        self._vector_to_entry: Dict[Tuple[int, ...], Dict[str, Any]] = {}
        self._id_to_entry: Dict[int, Dict[str, Any]] = {}
        self._filename_to_entry: Dict[str, Dict[str, Any]] = {}
        self._ensure_loaded()

    def _ensure_loaded(self):
        if self._meta is not None:
            return
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, "r") as f:
                self._meta = json.load(f)
            for entry in self._meta.get("states", []):
                vec = tuple(entry["vector"])  # [peg_smallest, peg_mid, peg_largest]
                self._vector_to_entry[vec] = entry
                self._id_to_entry[entry["id"]] = entry
                self._filename_to_entry[entry["filename"]] = entry
        else:
            # Fallback: build minimal index by parsing available filenames
            self._meta = {"n_disks": 3, "n_pegs": 3, "states": []}
            if os.path.isdir(self.images_dir):
                for name in os.listdir(self.images_dir):
                    if not name.endswith(".png"):
                        continue
                    m = re.match(r"s(\d+)_(?:.*)?d0(\d)d1(\d)d2(\d)\.png$", name)
                    if not m:
                        # Try simpler: sNN_*.png without vector; skip if unparseable
                        continue
                    sid = int(m.group(1))
                    vec = (int(m.group(2)), int(m.group(3)), int(m.group(4)))
                    entry = {"id": sid, "vector": list(vec), "filename": name, "moves": []}
                    self._meta["states"].append(entry)
                    self._vector_to_entry[vec] = entry
                    self._id_to_entry[sid] = entry
                    self._filename_to_entry[name] = entry

    @property
    def n_disks_supported(self) -> int:
        return int(self._meta.get("n_disks", 3))

    @property
    def n_pegs_supported(self) -> int:
        return int(self._meta.get("n_pegs", 3))

    def _validate_dims(self, num_disks: int):
        if num_disks != self.n_disks_supported or self.n_pegs_supported != 3:
            raise ValueError(
                f"ImageStateRepresentation supports exactly {self.n_disks_supported} disks and {self.n_pegs_supported} pegs; got {num_disks} disks."
            )

    def from_internal_state(self, internal_state: Tuple[int, ...], num_disks: int) -> Dict[str, Any]:
        self._validate_dims(num_disks)
        vec = tuple(int(x) for x in internal_state)
        entry = self._vector_to_entry.get(vec)
        if not entry:
            raise ValueError(f"No image found for state vector {vec} in {self.images_dir}")
        filename = entry["filename"]
        abs_path = os.path.join(self.images_dir, filename)
        return {
            "id": entry.get("id"),
            "vector": list(vec),
            "filename": filename,
            "path": abs_path,
        }

    def to_internal_state(self, custom_state: Any, num_disks: int) -> Tuple[int, ...]:
        self._validate_dims(num_disks)
        entry = None
        if isinstance(custom_state, dict):
            if "vector" in custom_state:
                vec = tuple(int(x) for x in custom_state["vector"])
                entry = self._vector_to_entry.get(vec)
            if entry is None and "id" in custom_state:
                entry = self._id_to_entry.get(int(custom_state["id"]))
            if entry is None and "filename" in custom_state:
                entry = self._filename_to_entry.get(str(custom_state["filename"]))
            if entry is None and "path" in custom_state:
                entry = self._filename_to_entry.get(os.path.basename(str(custom_state["path"])))
        elif isinstance(custom_state, int):
            entry = self._id_to_entry.get(custom_state)
        elif isinstance(custom_state, str):
            # Could be filename or path
            entry = self._filename_to_entry.get(os.path.basename(custom_state))

        if not entry:
            raise ValueError(
                "Could not resolve image state back to an internal vector. Provide dict with 'id'/'filename'/'path' or the vector itself."
            )

        vec = tuple(int(x) for x in entry["vector"])
        return vec

    def describe(self, custom_state: Any) -> str:
        try:
            if isinstance(custom_state, dict):
                fid = custom_state.get("id")
                fname = custom_state.get("filename")
                path = custom_state.get("path")
                return f"Image: {fname} (id {fid}) at {path}"
            elif isinstance(custom_state, str):
                return f"Image file: {os.path.basename(custom_state)}"
            elif isinstance(custom_state, int):
                return f"Image id: {custom_state}"
        except Exception:
            pass
        return str(custom_state)


# Registry for easy access to representations
REPRESENTATIONS = {
    "default": DefaultStateRepresentation,
    "dict_list": DictListStateRepresentation,
    "matrix": MatrixStateRepresentation,
    "natural_language": NaturalLanguageStateRepresentation,
    "lua_function": LuaFunctionStateRepresentation,
    "image": ImageStateRepresentation,
    "image_state": ImageStateRepresentation,
}


def get_representation(name: str) -> StateRepresentation:
    """Factory function to get representation by name"""
    if name not in REPRESENTATIONS:
        raise ValueError(
            f"Unknown representation: {name}. Available: {list(REPRESENTATIONS.keys())}"
        )
    return REPRESENTATIONS[name]()
