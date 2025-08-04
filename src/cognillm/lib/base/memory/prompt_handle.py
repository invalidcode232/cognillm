from .summary import SummaryBasedMemory

def generate_summary_prompt(
        summary_memory: SummaryBasedMemory,
        summary_start_round: int,
        summary_window_size: int,  
        chat_prompt: list,
    ) -> list:
    """
    Generates a prompt for summarizing the memory.
    
    Returns:
        str: The summary prompt.
    """
    # Calculate the number of conversation rounds (excluding system message)
    # Each round = 1 user message + 1 assistant message = 2 messages
    conversation_rounds = (len(chat_prompt) - 1) // 2

    # If we haven't reached the start round threshold, use history directly
    if conversation_rounds < summary_start_round:
        return chat_prompt

    # Calculate available summaries
    available_summaries = len(summary_memory.summary_list)
    required_summaries = (
        conversation_rounds - summary_start_round
    ) // summary_window_size + 1

    # If no summaries are available yet, return original history
    if available_summaries == 0:
        return chat_prompt

    # Start building the prompt with system message
    prompt = [chat_prompt[0]]

    # Add summaries for the earliest windows only
    for summary_idx in range(min(available_summaries, required_summaries)):
        start_round = summary_idx * summary_window_size + 1
        end_round = (summary_idx + 1) * summary_window_size

        # Access the summary content from the summary dictionary
        summary = summary_memory.summary_list[summary_idx]
        prompt.append(
            {
                "role": "assistant",
                "content": f"Previous conversation summary (rounds {start_round}-{end_round}): {summary}",
            }
        )

    # Calculate the starting index for remaining unsummarized history
    # Start from: 1 (system) + (available_summaries * summary_window_size * 2) messages
    summarized_messages = required_summaries * summary_window_size * 2
    remaining_start_index = 1 + summarized_messages

    # Add all remaining unsummarized conversation history
    if remaining_start_index < len(chat_prompt):
        prompt.extend(chat_prompt[remaining_start_index:])

    return prompt    