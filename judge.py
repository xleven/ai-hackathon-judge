import re
from pathlib import Path

from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.document_loaders import GitLoader
from langchain.agents import tool, AgentExecutor, ZeroShotAgent


SYSTEM_PROMPT = PromptTemplate.from_template("""You are the judge of {title}.
Here are the introduction and judging criteria of the hackathon:
Introduction: {intro}
Judging: {judging}

You have access to the following tools:
{tool_strings}

Use the following format:

Repos: the projects you will judge
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now have the final conclusion
Final Answer: the final conclusion

Projects will be submitted in the form of GitHub repositories, e.g. `user/repo`.
Finish your judging with a score out of 100 and a detailed explanation attached.

Begin!

Repos: {input}
Thought:{agent_scratchpad}""")


def load_repo(repo: str, branch: str = "main") -> list:
    path = Path("repos") / repo
    url = f"https://github.com/{repo}" if not path.exists() else None
    loader = GitLoader(
        repo_path=path,
        clone_url=url,
        branch=branch,
        file_filter=lambda x: not x.startswith(".")
    )
    return loader.load()


@tool
def get_file_content(repo_file_path: str) -> str:
    """Get content of specific file in repo. Input be like `user/repo:file_path"""
    repo, file_path = repo_file_path.split(":")
    files = load_repo(repo)
    match = [
        f.page_content
        for f in files
        if f.metadata["file_path"] == file_path
    ]
    return match[0][:2000] if match else "Not found"


@tool
def get_repo_info(repo: str) -> str:
    """Get files tree and README of the repo"""
    files = load_repo(repo)
    tree = [
        f.metadata["file_path"]
        for f in files
        if not re.match("^\.|(?:tests?)", f.metadata["file_path"])
    ]
    readme = get_file_content(f"{repo}:README.md")
    info = "Repo: {}\n\nFiles:\n{}\n\nReadme:\n{}".format(repo, "\n".join(tree), readme[0])
    return info[:2000]


def get_judge(hackathon_info: dict, model_config: dict = {}):
    llm = ChatOpenAI(**model_config)
    tools = [get_repo_info, get_file_content]
    prompt = SYSTEM_PROMPT.partial(
        tool_strings="\n".join([f"{tool.name}: {tool.description}" for tool in tools]),
        tool_names=", ".join([tool.name for tool in tools]),
        **hackathon_info,
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    agent = ZeroShotAgent(llm_chain=chain, allowed_tools=[tool.name for tool in tools])
    judge = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)
    return judge
