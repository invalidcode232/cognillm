import os


class PromptManager:
    def __init__(
        self,
        brain_prompt_name: str,
        messagegen_prompt_name: str,
        cognitive_model: str,
        profile_name: str,
    ):
        brain_prompt_path = os.path.join(
            os.path.dirname(__file__), "include", "prompts", f"{brain_prompt_name}"
        )
        messagegen_prompt_path = os.path.join(
            os.path.dirname(__file__),
            "include",
            "prompts",
            f"{messagegen_prompt_name}",
        )

        with open(brain_prompt_path, "r") as file:
            self.brain_prompt = (
                file.read()
                .replace("%name%", profile_name)
                .replace("%cognitive_model%", cognitive_model)
            )

        with open(messagegen_prompt_path, "r") as file:
            self.messagegen_prompt = file.read().replace("%name%", profile_name)

    def get_brain_prompt(self):
        return self.brain_prompt

    def get_textgen_prompt(self):
        return self.messagegen_prompt
