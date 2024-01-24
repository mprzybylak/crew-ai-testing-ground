from crewai import Agent, Task, Crew
from langchain.tools import DuckDuckGoSearchRun

search_tool = DuckDuckGoSearchRun()

researcher = Agent(
    role='Researcher',
    goal='Searching internet to find relevant info',
    backstory='''
    You have many years of experience doing research on internet.
    ''',
    verbose=True,
    allow_delegation=False,
    tools=[search_tool]
)

task = Task(
    description='Find information about best trombone brands',
    agent=researcher
)

crew = Crew(
    agents=[researcher],
    tasks=[task],
    verbose=2
)

result = crew.kickoff()

print("##########")
print(result)