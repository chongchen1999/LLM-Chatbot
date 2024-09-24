from query_sender import send_query

def write_result_to_file(model_name, result, filename):
    """
    Writes the result of a query to a file.

    Parameters:
        model_name (str): The name of the model.
        result (dict): The result of the query containing metrics and generated text.
        filename (str): The name of the file to write the result to.
    """
    filename = f"test_results/{filename}"
    with open(filename, "w") as f:
        f.write(f"Total Tokens: {result['total_tokens']}\n")
        f.write(f"Total Time: {result['total_time']:.4f} seconds\n")
        if result['time_to_first_token']:
            f.write(f"Time to generate first token: {result['time_to_first_token']:.4f} seconds\n")
        if result['avg_token_latency']:
            f.write(f"Average Token Latency: {result['avg_token_latency']:.4f} ms/token\n")
        f.write("\nFinal Generated Text:\n")
        f.write(result['result'])

def run_simple_query(model_name):
    """Runs a simple query and writes the result to a file."""
    prompt_simple = "What is the capital of France?"
    result_simple = send_query(model_name, prompt_simple)
    write_result_to_file(model_name, result_simple, f"{model_name}_simple_query_result.txt")

def run_multi_turn_conversation(model_name):
    """Runs a multi-turn conversation and writes both responses to a single file."""
    
    # Step 1: Initialize the conversation with the system message
    conversation = [{"role": "system", "content": "You are a helpful assistant."}]
    
    # Step 2: Send the first prompt ("Who are you?")
    prompt1_conversation = "Who are you?"
    result_conversation_1 = send_query(model_name, prompt1_conversation, conversation=conversation)
    
    # Append the first interaction to the conversation history
    conversation.append({"role": "user", "content": prompt1_conversation})
    conversation.append({"role": "assistant", "content": result_conversation_1["result"]})
    
    # Step 3: Send the second prompt ("Who is the most powerful LLM model in the world?")
    prompt2_conversation = "Who is the most powerful LLM model in the world?"
    result_conversation_2 = send_query(model_name, prompt2_conversation, conversation=conversation)
    
    # Step 4: Combine both results for file writing
    combined_result = {
        "total_tokens": result_conversation_1["total_tokens"] + result_conversation_2["total_tokens"],
        "total_time": result_conversation_1["total_time"] + result_conversation_2["total_time"],
        "time_to_first_token": result_conversation_1["time_to_first_token"],
        "avg_token_latency": (
            (result_conversation_1["avg_token_latency"] + result_conversation_2["avg_token_latency"]) / 2
            if result_conversation_1["avg_token_latency"] and result_conversation_2["avg_token_latency"]
            else None
        ),
        "result": (
            "First response:\n" + result_conversation_1["result"] + "\n\n" +
            "Second response:\n" + result_conversation_2["result"]
        )
    }
    
    # Step 5: Write the combined result to a single file
    write_result_to_file(model_name, combined_result, f"{model_name}_multi_turn_result.txt")


def run_parameter_experiments(model_name):
    """Runs experiments with different parameters and writes the results to files."""
    temperatures = [0.5, 1.0]  # Different temperature values
    max_tokens_list = [50, 100]  # Different max tokens values

    for temperature in temperatures:
        for max_tokens in max_tokens_list:
            prompt_experiment = "Write a short poem about the stars."
            result_experiment = send_query(model_name, prompt_experiment, temperature=temperature, max_tokens=max_tokens)
            write_result_to_file(model_name, result_experiment, f"{model_name}_experiment_t{temperature}_m{max_tokens}.txt")

def run_specific_task(model_name):
    """Runs a specific task and writes the result to a file."""
    prompt_task = "Explain the theory of relativity."
    result_task = send_query(model_name, prompt_task)
    write_result_to_file(model_name, result_task, f"{model_name}_task_result.txt")

def run_experiments(model_name):
    """Runs various experiments for the given model."""
    run_simple_query(model_name)
    run_multi_turn_conversation(model_name)
    run_parameter_experiments(model_name)
    run_specific_task(model_name)

    print(f"Results for {model_name} written to respective files.")


if __name__ == "__main__":
    models_to_test = ["llama3", "gemma2"]  # List of models to test
    
    for model in models_to_test:
        print(f"Running experiments for {model}...")
        run_experiments(model)
