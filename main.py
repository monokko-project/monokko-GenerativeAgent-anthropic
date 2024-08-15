import dotenv
import os
# from agenet import anthropic_model
import warnings
from agent.anthropic_model import Monokko
import json
import pathlib
warnings.simplefilter('ignore')

def _set_up():
    dotenv.load_dotenv()
    if os.getenv("ANTHROPIC_API_KEY") is None:
        raise ValueError("ANTHROPIC_API_KEY is not set.")

def load_monokko_configs(dir_path="monokkoConfigs"):
    config_list = []

    config_path = pathlib.Path(__file__).parent / dir_path
    for config_file in config_path.iterdir():
        if config_file.suffix == ".json":
            with open(config_file, "r") as f:
                config_list.append(json.load(f))

    return config_list


def run_conversation(agents: dict, agent_keys: list,  initial_observation: str) -> None:
    """Runs a conversation between agents."""
    # _, observation = agents[1].generate_reaction(initial_observation)
    _, observation = agents[agent_keys[0]].generate_dialogue_response(initial_observation)

    # print(observation)
    turns = 0

    # return
    while True:
        break_dialogue = False
        for agent_name, agent_inst in agents.items():
            stay_in_dialogue, observation = agent_inst.generate_dialogue_response(
                observation
            )
            print(observation)
            # observation = f"{agent.name} said {reaction}"
            if not stay_in_dialogue:
                break_dialogue = True
        if break_dialogue:
            break
        turns += 1


def main():

    monokko_configs = load_monokko_configs()
    mono = Monokko()

    ### Create agents

    agent_names = []

    for config in monokko_configs:
        agent_names.append(config["name"])
        mono.create_agent(config["name"], agent_traits=config["traits"], agent_status=config["status"])
        # print(mono.agents[config["name"]].get_summary())
        mono.agents[config["name"]].add_observations(config["observations"])
        print(mono.agents[config["name"]].get_summary())

    ### Interact with agents

    names_text = ", ".join(agent_names)
    observation = f"こんにちは！調子はどう？他のエージェントたち（{names_text}）にはなしかけてみたら？ {agent_names[-1]}がなにか言ってたよ。"

    run_conversation(mono.agents, agent_names, observation)



if __name__ == "__main__":
    _set_up()
    main()

