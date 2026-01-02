"""
Test script to verify state representations work correctly
"""


def test_basic_conversions():
    """Test basic conversion functionality for all representations"""
    from libs.smartplay.src.smartplay.hanoi.state_representations import (
        DefaultStateRepresentation,
        DictListStateRepresentation,
        MatrixStateRepresentation,
        NaturalLanguageStateRepresentation,
        get_representation,
    )

    print("=== TESTING BASIC CONVERSIONS ===")

    # Test cases: (internal_state, num_disks, description)
    test_cases = [
        ((0, 0, 0), 3, "All disks on peg A"),
        ((2, 2, 2), 3, "All disks on peg C (goal state)"),
        ((0, 1, 2), 3, "Disk 0 on A, disk 1 on B, disk 2 on C"),
        ((2, 1, 2), 3, "Disks 0,2 on C, disk 1 on B"),
        ((0, 0), 2, "2-disk: all on A"),
        ((1, 0), 2, "2-disk: disk 0 on B, disk 1 on A"),
    ]

    representations = [
        ("default", DefaultStateRepresentation()),
        ("dict_list", DictListStateRepresentation()),
        ("matrix", MatrixStateRepresentation()),
        ("natural_language", NaturalLanguageStateRepresentation()),
    ]

    for rep_name, rep_obj in representations:
        print(f"\n--- Testing {rep_name.upper()} representation ---")

        for internal_state, num_disks, description in test_cases:
            try:
                # Convert to external format
                external_state = rep_obj.from_internal_state(internal_state, num_disks)

                # Convert back to internal format
                converted_back = rep_obj.to_internal_state(external_state, num_disks)

                # Verify round-trip conversion
                success = converted_back == internal_state
                status = "PASS" if success else "FAIL"

                print(f"{status} {description}")
                print(f"   Internal: {internal_state}")
                print(f"   External: {external_state}")
                print(f"   Back:     {converted_back}")
                print(f"   Describe: {rep_obj.describe(external_state)}")

                if not success:
                    print(f"   ERROR: Round-trip conversion failed!")

            except Exception as e:
                print(f"FAIL {description} - ERROR: {e}")

            print()


def test_environments():
    """Test the new environment classes"""
    from libs.smartplay.src.smartplay.hanoi.hanoi_env import (
        Hanoi3Disk,
        Hanoi3DiskDictList,
        Hanoi3DiskMatrix,
        Hanoi3DiskNaturalLanguage,
        Hanoi3DiskLuaFunction,
    )

    print("\n=== TESTING ENVIRONMENT CLASSES ===")

    environments = [
        ("Hanoi3Disk (default)", Hanoi3Disk),
        ("Hanoi3DiskDictList", Hanoi3DiskDictList),
        ("Hanoi3DiskMatrix", Hanoi3DiskMatrix),
        ("Hanoi3DiskNaturalLanguage", Hanoi3DiskNaturalLanguage),
        ("Hanoi3DiskLuaFunction", Hanoi3DiskLuaFunction),
    ]

    for env_name, env_class in environments:
        print(f"\n--- Testing {env_name} ---")

        try:
            # Create environment
            env = env_class()

            # Reset environment
            state, info = env.reset()

            print(f"PASS Environment created and reset successfully")
            print(f"Internal state: {state}")
            print(f"Formatted state: {info['formatted_state']}")
            print(f"Description:\n{info['obs']}")

            # Test a few moves
            valid_actions = env.get_valid_actions()
            if valid_actions:
                action = valid_actions[0]
                print(f"\nTesting action {action}:")

                next_state, reward, done, step_info = env.step(action)
                print(f"PASS Step executed successfully")
                print(f"New internal state: {next_state}")
                print(f"New formatted state: {step_info['formatted_state']}")
                print(f"Reward: {reward}, Done: {done}")

        except Exception as e:
            print(f"FAIL Error testing {env_name}: {e}")
            import traceback

            traceback.print_exc()


def test_representation_switching():
    """Test switching representations at runtime"""
    from libs.smartplay.src.smartplay.hanoi.hanoi_env import Hanoi3Disk

    print("\n=== TESTING REPRESENTATION SWITCHING ===")

    try:
        # Create environment with default representation
        env = Hanoi3Disk()
        state, info = env.reset()

        representations = ["default", "dict_list", "matrix", "natural_language"]

        for rep_name in representations:
            print(f"\n--- Switching to {rep_name} ---")

            # Switch representation
            env.set_state_representation(rep_name)

            # Get formatted state
            formatted_state = env.get_formatted_state()
            print(f"Formatted state: {formatted_state}")
            print(f"Description: {env.state_formatter.describe(formatted_state)}")

    except Exception as e:
        print(f"FAIL Error in representation switching: {e}")
        import traceback

        traceback.print_exc()


def test_factory_function():
    """Test the get_representation factory function"""
    from libs.smartplay.src.smartplay.hanoi.state_representations import (
        get_representation,
        REPRESENTATIONS,
    )

    print("\n=== TESTING FACTORY FUNCTION ===")

    # Test valid representations
    for rep_name in REPRESENTATIONS.keys():
        try:
            rep_obj = get_representation(rep_name)
            print(
                f"PASS Successfully created {rep_name} representation: {type(rep_obj).__name__}"
            )
        except Exception as e:
            print(f"FAIL Error creating {rep_name} representation: {e}")

    # Test invalid representation
    try:
        invalid_rep = get_representation("invalid_name")
        print(f"FAIL Should have failed for invalid representation name")
    except ValueError as e:
        print(f"PASS Correctly raised error for invalid name: {e}")
    except Exception as e:
        print(f"FAIL Unexpected error for invalid name: {e}")


def test_edge_cases():
    """Test edge cases and error conditions"""
    from libs.smartplay.src.smartplay.hanoi.state_representations import (
        DictListStateRepresentation,
        MatrixStateRepresentation,
    )

    print("\n=== TESTING EDGE CASES ===")

    dict_rep = DictListStateRepresentation()
    matrix_rep = MatrixStateRepresentation()

    # Test invalid states
    test_cases = [
        ("Invalid disk ID in dict", {"A": [5], "B": [], "C": []}, 3, dict_rep),
        ("Missing disk in dict", {"A": [1], "B": [], "C": []}, 3, dict_rep),
        ("Duplicate disk in dict", {"A": [1], "B": [1], "C": []}, 3, dict_rep),
        ("Wrong matrix dimensions", [[1, 2]], 3, matrix_rep),
        (
            "Invalid disk in matrix",
            [[5, -1, -1], [-1, -1, -1], [-1, -1, -1]],
            3,
            matrix_rep,
        ),
    ]

    for description, invalid_state, num_disks, rep_obj in test_cases:
        try:
            result = rep_obj.to_internal_state(invalid_state, num_disks)
            print(f"FAIL {description} - Should have failed but got: {result}")
        except (ValueError, KeyError) as e:
            print(f"PASS {description} - Correctly raised error: {type(e).__name__}")
        except Exception as e:
            print(f"? {description} - Unexpected error type: {type(e).__name__}: {e}")


def test_game_sequence():
    """Test a complete game sequence"""
    from libs.smartplay.src.smartplay.hanoi.hanoi_env import Hanoi3DiskDictList

    print("\n=== TESTING COMPLETE GAME SEQUENCE ===")

    try:
        env = Hanoi3DiskDictList()
        state, info = env.reset()

        print(f"Initial state: {info['formatted_state']}")

        # Try to solve 3-disk Hanoi (optimal solution: 7 moves)
        # Move sequence for 3 disks: A->C, A->B, C->B, A->C, B->A, B->C, A->C
        optimal_moves = [
            1,
            0,
            4,
            1,
            2,
            5,
            1,
        ]  # Actions corresponding to the moves above

        for i, action in enumerate(optimal_moves):
            print(f"\nMove {i+1}: Action {action}")

            valid_actions = env.get_valid_actions()
            if action not in valid_actions:
                print(f"FAIL Action {action} not in valid actions: {valid_actions}")
                break

            state, reward, done, info = env.step(action)
            print(f"State: {info['formatted_state']}")
            print(f"Reward: {reward}, Done: {done}")

            if done:
                print(f"PASS Game completed in {i+1} moves!")
                break
        else:
            print(f"FAIL Game not completed after {len(optimal_moves)} moves")

    except Exception as e:
        print(f"FAIL Error in game sequence: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    """Run all tests"""
    print("Testing State Representations for Hanoi Environment")
    print("=" * 60)

    test_basic_conversions()
    test_environments()
    test_representation_switching()
    test_factory_function()
    test_edge_cases()
    test_game_sequence()

    print("\n" + "=" * 60)
    print("Testing completed!")
