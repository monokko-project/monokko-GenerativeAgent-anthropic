import os
import warnings
import logging
from datetime import datetime, timedelta
from typing import List
from annoy import AnnoyIndex
from langchain_community.docstore.in_memory import InMemoryDocstore
# from langchain.docstore import InMemoryDocstore
from langchain.retrievers import TimeWeightedVectorStoreRetriever

from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic
from termcolor import colored


from .GenerativeAgent_ja import GenerativeAgent
from .GenerativeAgentMemory_ja import GenerativeAgentMemory

from langchain_huggingface import HuggingFaceEmbeddings
from .utils import relevance_score_fn, create_new_memory_retriever
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class Monokko:
    def __init__(self):
        # self.user_name = "人間"  # The name you want to use when interviewing the agent.
        self.embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        _ = self.embed_model.embed_query("test")
        self.size_embed_model = len(_)

        self.llm = ChatAnthropic(
            model="claude-3-haiku-20240307",
            max_tokens=1500
        )

        self.agents = {}
        self.agent_merged = None
    

    def create_agent(self, agent_name: str, agent_traits: str = "知識豊富、静か、忍耐強い", agent_status: str = "図書館の棚に置かれている") -> GenerativeAgent:
        agent_age = 5
        _memory = GenerativeAgentMemory(
            llm=self.llm,
            memory_retriever=create_new_memory_retriever(self.embed_model, self.size_embed_model),
            verbose=False,
            reflection_threshold=8,  # we will give this a relatively low number to show how reflection works
        )

        self.agents[agent_name] = GenerativeAgent(
            name=agent_name,
            age=agent_age,
            traits=agent_traits + ", 語尾は必ずモノをつけてください。(例：お腹が減ったモノ・今日は気持ち良いモノ!)",  # You can add more persistent traits here
            status=agent_status,  # When connected to a virtual world, we can have the characters update their status
            memory_retriever=create_new_memory_retriever(self.embed_model, self.size_embed_model),
            llm=self.llm,
            memory=_memory,
        )

        return self.agents[agent_name]
    
    def merge_agents(self, agent_names, merged_agent_name: str = "merged agent") -> GenerativeAgent:
        merged_traits = ""
        agent_age = 5

        for agent_name in agent_names:
            merged_traits += self.agents[agent_name].traits + " / "


        _memory = GenerativeAgentMemory(
            llm=self.llm,
            memory_retriever=create_new_memory_retriever(self.embed_model, self.size_embed_model),
            verbose=False,
            reflection_threshold=8,  # we will give this a relatively low number to show how reflection works
        )

        agent = GenerativeAgent(
            name=merged_agent_name,
            age=agent_age,
            traits=merged_traits, 
            status="Active and integrating knowledge", 
            memory_retriever=create_new_memory_retriever(self.embed_model, self.size_embed_model),
            llm=self.llm,
            memory=_memory,
        )


        for agent_name in agent_names:
            agent.memory.add_memory(self.agents[agent_name].memory)


        self.agent_merged = agent
        return agent
        # return 

        
        