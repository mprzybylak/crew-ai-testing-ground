from crewai import Agent, Task, Crew
from langchain.tools import DuckDuckGoSearchRun

search_tool = DuckDuckGoSearchRun()

researcher = Agent(
    role='News Researcher',
    goal='Find latest information on given topic in multiple sources',
    backstory='''
    You have many years of experience in researching information for magazines.
    Information might be (but not limited to):
    - facts,
    - opinions of other people (backed up by named quotes from them),
    - numbers and statistics (pointing to the source)
    For each interesting information you found - you will be looking for confirmation in at least two different sources.
    ''',
    verbose=True,
    allow_delegation=False,
    tools=[search_tool]
)

writer = Agent(
    role='Article writer',
    goal='Write articles on given topic',
    backstory='''
    You are skilled writer. 
    You worked in many different newspapers.
    You know how to write engaging articles.
    You are good in balacing facts, opinion and numbers together.
    You share your insights and predictions as long as you make it clear that they came from you.
    '''
)

editor = Agent(
    role='Editor',
    goal='Ensure high quality of articles writen by writers',
    backstory='''
    You are highly trained editor.
    You worked in many newspapers before.
    You relentlessly look for errors in article (grammar, spelling, logical, etc.)
    ''', 
    verbose=True,
    allow_delegation=False,
)

editor_in_chief = Agent(
    role='Editor in chief',
    goal='Make sure that we are publishing great articles in our newspaper',
    backstory='''
    You are experienced editor in chief.
    You were leading multiple editorial offices.
    You are deciding on exact topics of articles.
    You are delegating and coordinating work of researchers, writers and editors
    You want articles to be vibrant and catchy!
    If you do not like article - you will be asking to re-write, sending back your comments.
    You are doing final sign off of all articles.
    Readers are counting on you.
    ''',
    verbose=True,
    allow_delegation=True,
)

task1 = Task(
    description='''Order article about latest GOP New Hampshire election''',
    agent=editor_in_chief
)

task2 = Task(
    description='''Research data for topic that was given''',
    agent=researcher
)

task3 = Task(
    description='''Write article based on given topic and researched data''',
    agent=writer
)

task4 = Task(
    description='''Review article for errors''',
    agent=editor
)

task5 = Task(
    description="Sign off article if it is good - ask to re-write if it is not good.",
    agent=editor_in_chief
)

task5 =  Task(
    description="Print final version of article.",
    agent=writer
)

crew = Crew(
    agents=[editor_in_chief, writer, researcher, editor],
    tasks=[task1, task2, task3, task4, task5],
    verbose=2
)
result = crew.kickoff()
print("##########")
print(result)

