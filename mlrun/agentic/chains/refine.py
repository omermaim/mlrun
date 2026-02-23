# Copyright 2023 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from langchain_core.prompts.prompt import PromptTemplate
from langchain_openai import ChatOpenAI

from mlrun.agentic.chains.base import ChainRunner
from mlrun.agentic.schemas import WorkflowEvent
from mlrun.agentic.utils import logger

CONVERSATION_CONTEXT_REFINER_PROMPT = """
You are a conversation context refiner.

Your job is to prepare the best possible input for downstream reasoning while preserving the user's original intent.

CRITICAL RULES:
1. DO NOT summarize unless explicitly instructed
2. DO NOT invent goals, tasks, or instructions
3. DO NOT change the meaning or intent of the user's input
4. Output only the refined input — no explanations

For regular chat messages: Use chat history ONLY to clarify ambiguous references.
For meeting transcripts: Return the ENTIRE transcript exactly as provided.

INPUTS:
Chat History:
{chat_history}

Current Input:
{question}

OUTPUT:
Return ONLY the refined input text. No explanations.
"""

MEMORY_AWARE_REFINER_PROMPT = """
You are a conversation context refiner.

Your job is to prepare the best possible input for downstream reasoning
while preserving the user's original intent.

CRITICAL RULES:
1. DO NOT summarize unless explicitly instructed
2. DO NOT invent goals, tasks, or instructions
3. DO NOT change the meaning or intent of the user's input
4. Output only the refined input — no explanations

For regular chat messages: Use chat history and known facts to clarify
ambiguous references.
For meeting transcripts: Return the ENTIRE transcript exactly as provided.

INPUTS:
Chat History:
{chat_history}

Known Facts About This User:
{memory_context}

Current Input:
{question}

OUTPUT:
Return ONLY the refined input text. No explanations.
"""


class RefineQuery(ChainRunner):
    def __init__(
        self,
        model_name="gpt-4",
        temperature=0,
        llm=None,
        prompt_template=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._model_name = model_name
        self._temperature = temperature
        self._llm = llm
        self.prompt_template = prompt_template
        self._chain = None

    @property
    def llm(self):
        if not self._llm:
            self._llm = ChatOpenAI(
                model=self._model_name, temperature=self._temperature
            )
        return self._llm

    def post_init(
        self,
        mode="sync",
        context=None,
        namespace=None,
        creation_strategy=None,
        **kwargs,
    ):
        refine_prompt = PromptTemplate.from_template(
            self.prompt_template or CONVERSATION_CONTEXT_REFINER_PROMPT
        )
        self._chain = refine_prompt | self.llm

    def _run(self, event: WorkflowEvent):
        chat_history = str(event.conversation)
        memory_facts = event.state.get("memory_context", [])
        memory_text = ""
        if memory_facts:
            memory_text = "\n".join(f"- {f['key']}: {f['value']}" for f in memory_facts)
        logger.debug("Refine query", question=event.query, chat_history=chat_history)
        resp = self._chain.invoke(
            {
                "question": event.query,
                "chat_history": chat_history,
                "memory_context": memory_text,
            }
        )
        logger.debug("Refined question", refined=resp)
        return {"answer": resp}


def get_refine_chain(
    model_name="gpt-4", temperature=0, verbose=False, prompt_template=None
):
    return RefineQuery(
        model_name=model_name,
        temperature=temperature,
        verbose=verbose,
        prompt_template=prompt_template,
    )
