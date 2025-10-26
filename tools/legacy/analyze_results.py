import pickle
import sys
import pandas as pd

def analyze_router_results(collection, router_type):
    router_results = None
    for result in collection.evaluation_results:
        # The router_type in the DefaultEvaluationResult is just the base name (e.g., 'compitum_router')
        # We need to match the full name from the per_prompt_results_dict keys
        if result.router_type.startswith(router_type): # Use startswith to match base name
            router_results = result
            break
    
    if router_results:
        print(f"\n--- Analysis for {router_type} ---")
        per_prompt_df = router_results.per_prompt_results
        if per_prompt_df is not None:
            model_counts = per_prompt_df['chosen_model'].value_counts(normalize=True)
            print(f"Model Selection Profile for {router_type}:")
            print(model_counts.to_string())

            # --- Error Analysis ---
            print(f"\n--- Error Analysis for {router_type} ---")
            # Compare chosen_model with oracle_chosen_model
            per_prompt_df['is_correct'] = (per_prompt_df['chosen_model'] == per_prompt_df['oracle_chosen_model'])
            
            incorrect_decisions = per_prompt_df[per_prompt_df['is_correct'] == False]
            
            print(f"Total prompts: {len(per_prompt_df)}")
            print(f"Correct decisions: {len(per_prompt_df[per_prompt_df['is_correct'] == True])}")
            print(f"Incorrect decisions: {len(incorrect_decisions)}")
            print(f"Accuracy (based on oracle match): {len(per_prompt_df[per_prompt_df['is_correct'] == True]) / len(per_prompt_df):.4f}")

            if not incorrect_decisions.empty:
                print("\nExamples of Incorrect Decisions:")
                # Print a few examples of incorrect decisions
                print(incorrect_decisions[['sample_id', 'prompt', 'chosen_model', 'oracle_chosen_model']].head().to_string())
            else:
                print(f"No incorrect decisions found for {router_type}.")

        else:
            print(f"Per-prompt results not available for {router_type}.")
    else:
        print(f"Could not find results for '{router_type}' in the specified file.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python analyze_results.py <file_path>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    try:
        with open(file_path, 'rb') as f:
            sys.path.insert(0, 'C:/Users/paulc/projects/compitum/src')
            collection = pickle.load(f)
        
        analyze_router_results(collection, 'compitum_router')
        analyze_router_results(collection, 'svm') # Analyze SVM as well
        analyze_router_results(collection, 'knn') # Analyze KNN as well
        analyze_router_results(collection, 'mlp') # Analyze MLP as well

    except Exception as e:
        print(f"Error analyzing pickle file: {e}")