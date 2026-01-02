import gym
import smartplay


def test_hanoi_representations():
    """Test all Hanoi representation variants"""

    # List of representations to test
    representations = [
        "Hanoi3Disk",  # Default
        "Hanoi3DiskDictList",
        "Hanoi3DiskMatrix",
        "Hanoi3DiskNaturalLanguage",
        "Hanoi3DiskLuaFunction",
    ]

    print("Testing Hanoi Representations")
    print("=" * 50)

    for rep in representations:
        print(f"\nTesting {rep}")
        print("-" * 30)

        try:
            # Create environment
            env = gym.make(f"smartplay:{rep}-v0")

            # Reset environment
            obs, info = env.reset()

            # Print the observation
            print(f"Observation format:")
            print(info["obs"])
            print(f"\nPASS {rep} - SUCCESS")

            # Test one step - get valid actions first
            valid_actions = env.get_valid_actions()
            if valid_actions:
                action = valid_actions[0]
                obs, reward, done, info = env.step(action)
                print(f"\nAfter action {action}:")
                print(info["obs"][:300] + "..." if len(info["obs"]) > 300 else info["obs"])

            env.close()

        except Exception as e:
            print(f"FAIL {rep} - FAILED: {str(e)}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 50)
    print("Test Complete!")


def test_representation_differences():
    """Test that different representations actually produce different outputs"""

    print("\nTesting Representation Differences")
    print("=" * 50)

    # Create environments with different representations
    default_env = gym.make("smartplay:Hanoi3Disk-v0")
    dict_env = gym.make("smartplay:Hanoi3DiskDictList-v0")
    matrix_env = gym.make("smartplay:Hanoi3DiskMatrix-v0")
    natural_env = gym.make("smartplay:Hanoi3DiskNaturalLanguage-v0")
    lua_env = gym.make("smartplay:Hanoi3DiskLuaFunction-v0")

    # Reset all environments
    default_obs, default_info = default_env.reset()
    dict_obs, dict_info = dict_env.reset()
    matrix_obs, matrix_info = matrix_env.reset()
    natural_obs, natural_info = natural_env.reset()
    lua_obs, lua_info = lua_env.reset()

    print("Default Representation:")
    print(default_info["obs"])
    print("\n" + "-" * 30)

    print("DictList Representation:")
    print(dict_info["obs"])
    print("\n" + "-" * 30)

    print("Matrix Representation:")
    print(matrix_info["obs"])
    print("\n" + "-" * 30)

    print("Natural Language Representation:")
    print(natural_info["obs"])
    print("\n" + "-" * 30)

    print("Lua Function Representation:")
    print(lua_info["obs"])
    print("\n" + "-" * 30)

    # Check if all representations produce different outputs
    representations = [
        default_info["obs"],
        dict_info["obs"],
        matrix_info["obs"],
        natural_info["obs"],
        lua_info["obs"],
    ]

    # Check that each representation is different from the others
    all_different = True
    for i in range(len(representations)):
        for j in range(i + 1, len(representations)):
            if representations[i] == representations[j]:
                print(f"FAIL Representations {i} and {j} are identical!")
                all_different = False

    if all_different:
        print("PASS All representations produce different outputs!")
    else:
        print("FAIL Some representations produce identical outputs!")

    # Close environments
    default_env.close()
    dict_env.close()
    matrix_env.close()
    natural_env.close()
    lua_env.close()


def test_specific_representation(representation="Hanoi3DiskMatrix"):
    """Test a specific representation in detail"""

    print(f"\nDetailed Test: {representation}")
    print("=" * 50)

    try:
        env = gym.make(f"smartplay:{representation}-v0")
        obs, info = env.reset()

        print("Initial State:")
        print(info["obs"])
        print("\n" + "-" * 40)

        # Test multiple moves
        max_steps = 5
        for i in range(max_steps):
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                print("No valid actions available!")
                break

            action = valid_actions[0]
            print(f"Step {i+1}: Action {action}")
            obs, reward, done, info = env.step(action)
            print(f"Reward: {reward}, Done: {done}")
            print(f"Observation:\n{info['obs']}")
            print("-" * 40)

            if done:
                print("Episode finished!")
                break

        env.close()
        print(f"PASS {representation} detailed test completed!")

    except Exception as e:
        print(f"FAIL {representation} detailed test failed: {str(e)}")
        import traceback

        traceback.print_exc()


def test_game_sequence():
    """Test a complete game sequence with optimal moves"""

    print("\nTesting Complete Game Sequence")
    print("=" * 50)

    try:
        env = gym.make("smartplay:Hanoi3DiskDictList-v0")
        obs, info = env.reset()

        print(f"Initial state: {info['formatted_state']}")

        # Optimal solution for 3-disk Hanoi: 7 moves
        # Move sequence: A->C, A->B, C->B, A->C, B->A, B->C, A->C
        optimal_moves = [1, 0, 4, 1, 2, 5, 1]

        for i, action in enumerate(optimal_moves):
            print(f"\nMove {i+1}: Action {action}")

            valid_actions = env.get_valid_actions()
            if action not in valid_actions:
                print(f"FAIL Action {action} not in valid actions: {valid_actions}")
                break

            obs, reward, done, info = env.step(action)
            print(f"State: {info['formatted_state']}")
            print(f"Reward: {reward}, Done: {done}")

            if done:
                print(f"PASS Game completed in {i+1} moves!")
                break
        else:
            print(f"FAIL Game not completed after {len(optimal_moves)} moves")

        env.close()

    except Exception as e:
        print(f"FAIL Error in game sequence: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    # Run all tests
    test_hanoi_representations()
    test_representation_differences()
    test_specific_representation("Hanoi3DiskMatrix")
    test_specific_representation("Hanoi3DiskNaturalLanguage")
    test_specific_representation("Hanoi3DiskLuaFunction")
    test_game_sequence()
