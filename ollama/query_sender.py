import requests
import json
import time

def send_query(model_name, prompt, temperature=1.0, max_tokens=100, conversation=None):
    """
    Sends a query to the model and returns the response, along with timing and token statistics.
    
    Parameters:
        model_name (str): Name of the model to query.
        prompt (str): The initial query to send to the model.
        temperature (float): Sampling temperature (controls randomness).
        max_tokens (int): Maximum number of tokens the model can generate.
        conversation (list): Optional, a list of previous exchanges for multi-turn conversation.
        
    Returns:
        dict: Packed result including total tokens, timings, and the model's generated response.
    """
    
    url = 'http://localhost:11434/api/generate'  # API endpoint for querying the model
    payload = {
        "model": model_name,
        "prompt": prompt,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    # If there's a conversation, append the previous context to the payload
    if conversation:
        payload["conversation"] = conversation
    
    # Track timings
    start_time = time.time()
    first_token_time = None
    total_tokens = 0
    result = ""

    try:
        # Send request
        response = requests.post(url, json=payload, stream=True)
        response.raise_for_status()  # Raise an error for bad responses
        
        # Collect and process response
        for line in response.iter_lines():
            if line:
                json_data = json.loads(line.decode('utf-8'))
                token = json_data.get("response", "")
                
                # Track time for the first token
                if first_token_time is None:
                    first_token_time = time.time()

                result += token
                total_tokens += len(token.split())  # Basic token counting, can be replaced with model-specific method

    except requests.exceptions.RequestException as e:
        print(f"Error during API request: {e}")
        return {
            "total_tokens": 0,
            "total_time": 0,
            "time_to_first_token": None,
            "avg_token_latency": None,
            "result": f"Error: {e}"
        }

    # Calculate timings
    end_time = time.time()
    total_time = end_time - start_time
    time_to_first_token = first_token_time - start_time if first_token_time else None
    avg_token_latency = total_time / total_tokens if total_tokens > 0 else None
    
    # Pack the results into a dictionary
    result_data = {
        "total_tokens": total_tokens,
        "total_time": total_time,
        "time_to_first_token": time_to_first_token,
        "avg_token_latency": avg_token_latency * 1000 if avg_token_latency else None,  # Latency in ms
        "result": result
    }

    return result_data