import os
import warnings
import logging
from datetime import datetime, timedelta
from typing import List
from annoy import AnnoyIndex
from langchain_community.docstore.in_memory import InMemoryDocstore
# from langchain.docstore import InMemoryDocstore
from langchain.retrievers import TimeWeightedVectorStoreRetriever
# import faiss
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic
from termcolor import colored


from langchain_experimental.generative_agents import (
    # GenerativeAgent,
    GenerativeAgentMemory,
)
from .GenerativeAgent_ja import GenerativeAgent

import math
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
    

    def create_agent(self, agent_name: str, agent_traits: str = "知識豊富、静か、忍耐強い", agent_status: str = "図書館の棚に置かれている") -> GenerativeAgent:

        _memory = GenerativeAgentMemory(
            llm=self.llm,
            memory_retriever=create_new_memory_retriever(self.embed_model, self.size_embed_model),
            verbose=False,
            reflection_threshold=8,  # we will give this a relatively low number to show how reflection works
        )

        self.agents[agent_name] = GenerativeAgent(
            name=agent_name,
            age=10,
            traits=agent_traits,  # You can add more persistent traits here
            status=agent_status,  # When connected to a virtual world, we can have the characters update their status
            memory_retriever=create_new_memory_retriever(self.embed_model, self.size_embed_model),
            llm=self.llm,
            memory=_memory,
        )

        return self.agents[agent_name]

