import gym
import smartplay


def test_messenger_representations():
    """Test all messenger representation variants"""

    # List of representations to test
    representations = [
        "MessengerL1",  # Default
        "MessengerL1Coordinates",
        "MessengerL1NaturalLanguagePos",
        "MessengerL1Symbolic",
    ]

    print("Testing Messenger Representations")
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

            # Test one step
            action = 0  # Move North
            obs, reward, done, info = env.step(action)
            print(f"\nAfter action (Move North):")
            print(info["obs"][:200] + "..." if len(info["obs"]) > 200 else info["obs"])

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
    default_env = gym.make("smartplay:MessengerL1-v0")
    matrix_env = gym.make("smartplay:MessengerL1Coordinates-v0")
    natural_env = gym.make("smartplay:MessengerL1NaturalLanguagePos-v0")
    symbolic_env = gym.make("smartplay:MessengerL1Symbolic-v0")

    # Reset all environments with same seed for consistency
    default_obs, default_info = default_env.reset()
    matrix_obs, matrix_info = matrix_env.reset()
    natural_obs, natural_info = natural_env.reset()

    print("Default Representation:")
    print(default_info["obs"])
    print("\n" + "-" * 30)

    print("Matrix Representation:")
    print(matrix_info["obs"])
    print("\n" + "-" * 30)
    
    print("Natural Language Representation:")
    print(natural_info["obs"])
    print("\n" + "-" * 30)
    


    # Check if all representations produce different outputs
    representations = [
        default_info["obs"], 
        matrix_info["obs"],
        natural_info["obs"],
    ]
    
    # Check that each representation is different from the others
    all_different = True
    for i in range(len(representations)):
        for j in range(i+1, len(representations)):
            if representations[i] == representations[j]:
                print(f"FAIL Representations {i} and {j} are identical!")
                all_different = False

    if all_different:
        print("PASS All representations produce different outputs!")
    else:
        print("FAIL Some representations produce identical outputs!")

    # Close environments
    default_env.close()
    matrix_env.close()
    natural_env.close()
    symbolic_env.close()


def test_specific_representation(representation="MessengerL1Coordinates"):
    """Test a specific representation in detail"""

    print(f"\nDetailed Test: {representation}")
    print("=" * 50)

    try:
        env = gym.make(f"smartplay:{representation}-v0")
        obs, info = env.reset()

        print("Initial State:")
        print(info["obs"])
        print("\n" + "-" * 40)

        # Test multiple actions
        actions = [0, 1, 2, 3, 4]  # North, South, West, East, Do Nothing
        action_names = [
            "Move North",
            "Move South",
            "Move West",
            "Move East",
            "Do Nothing",
        ]

        for i, (action, action_name) in enumerate(zip(actions, action_names)):
            print(f"Step {i+1}: {action_name}")
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




if __name__ == "__main__":
    # Run all tests
    test_messenger_representations()
    test_representation_differences()
    test_specific_representation("MessengerL1Coordinates")
    test_specific_representation("MessengerL1NaturalLanguagePos")
    test_specific_representation("MessengerL1Symbolic")
