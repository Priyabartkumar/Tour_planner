#First go through the instruction given in the txt file(installation)
from crewai import LLM
from langchain_ollama.llms import OllamaLLM

llm = LLM(
    model="llama3.2",
    base_url="http://localhost:11500"
)

from crewai.tools import tool
from langchain_community.tools import DuckDuckGoSearchResults
import json


@tool
def search_web_tool(query: str):
  """
  Searches the web and returns results.
  """

  search_tool = DuckDuckGoSearchResults(num_results=10, verbose=True)
  return search_tool.run(query)

from crewai import Agent
#Agent Researcher
guide_expert= Agent(
    role="City Local Guide Expert",
    goal="Provides information on things to do in the city based on the user's interests.",
    backstory="""A local expert with a passion for sharing the best experience and hidden gems of their city.""",
    tools=[search_web_tool],
    verbose=True,
    max_iter=5,
    llm=LLM(model="llama3.2",base_url="http://localhost:11500"),   #or use model="ollama/llama3.2"
    allow_delegation=False,
)

location_expert = Agent(
    role="Travel trip Expert",
    goal="Adapt to the user destination city language(French if city in French Country. Gather helpul information about the city )",
    backstory="""A seasoned traveler who has explored various destination and knows the ins and out of travel logistics.""",
    tools=[search_web_tool],
    verbose=True,
    max_iter=5,
    llm=LLM(model="ollama/llama3.2",base_url="http://localhost:11500"),
    allow_delegation=False,
)

planner_expert = Agent(
     role="Travel Planning Expert",
     goal="Compiles all gathered information to provide a comprehensive travel plan.",
     backstory="""
     Your are a professional guide with a passion for travel.
     An organized wizard who can turn a list of possibilities into a seamless itinerary.
     """,
     tools=[search_web_tool],
     verbose=True,
     max_iter=5,
     llm=LLM(model="ollama/llama3.2",base_url="http://localhost:11500"),
     allow_delegation=False,
)

from datetime import datetime
from crewai import Task

from_city= "India"
destination_city = "Rome"
date_from = "7th March 2025"
date_to = "14th march 2025"
interests = "sight seeing and good food" 

location_task = Task(
    description=f"""
    In French : This task involves a comprehensive data collection process to provide the traveler with essential information about their destination. It includes researching and compiling details on various accommodations, ranging from budget-friendly hostels to luxury hotels, as well as estimating the cost of living in the area. The task also covers transportation options, visa requirements, and any travel advisories that may be relevant.
    consider also the weather conditions forcast on the travel dates. and all the events that may be relevant to the traveler during the trip period.
    
    Traveling from : {from_city}
    Destination city : {destination_city}
    Arrival Date : {date_from}
    Departure Date : {date_to}

    Follow this rules :
    1. if the {destination_city} is in a French country : Respond in French.
    """,
    expected_output=f"""
    if the {destination_city} is in a French country : Respond in FRENCH.
    In markdown format : A detailed markdown report that includes a curated list
    """,
    agent=location_expert,
    output_file='city_report.md',
)

guide_task = Task(
    description=f"""
    if the {destination_city} is in a French country : Respond in FRENCH.
    Tailored to the traveler's personal ({interests}, this task focuses on creating an engaging and informative guide to the city's attractions. It involves identifying cultural landmarks, historical spots, entertainment venues, dining experiences, and outdoor activities that align with the user's preferences such {interests}. The guide also highlights seasonal events and festivals that might be of interest during the traveler's visit.
    Destination city : {destination_city}
    interests : {interests}
    Arrival Date : {date_from}
    Departure Date : {date_to}

    Follow this rule :
    1. if the {destination_city} is in a French country : Respond in FRENCH.
    """,

    expected_output=f"""
    An interactive markdown report that presents a personalized itinerary of activities and attractions, complete with descriptions, locations, and any necessary reservations or tickets.
    """,

    agent=guide_expert,
    output_file='guide_report.md',
)

planner_task = Task(
    description=f"""
    This task synthesizes all collected information into a detaileds introduction to the city (description of city and presentation, in 3 pragraphes) cohesive and practical travel plan. and takes into account the traveler's schedule, preferences, and budget to draft a day-by-day itinerary. The planner also provides insights into the city's layout and transportation system to facilitate easy navigation.
    Destination city : {destination_city}
    interests : {interests}
    Arrival Date : {date_from}
    Departure Date : {date_to}

    Follow this rules : 
    1. if the {destination_city} is in a French country : Respond in FRENCH.
    """,
    expected_output="""
    if the {destination_city} is in a French country : Respond in FRENCH.
    A rich markdown document with emojis on each title and subtitle, that :
    In markdown format : 
    # Welcome to {destination_city} :
    A 4 paragraphes markdown formated including :
    - a curated articles of presentation of the city, 
    - a breakdown of daily living expenses, and spots to visit.
    # Here's your Travel Plan to {destination_city} :
    Outlines a daily detailed travel plan list with time allocations and details for each activity, along with an overview of the city's highlights based on the guide's recommendations
    """,
    context=[location_task, guide_task],
    #context=context,
    agent=planner_expert,
      output_file='travel_plan.md',
)

# Task: Location
def location_task(agent, from_city, destination_city, date_from, date_to):
    return Task(
        description=f"""
        In French : This task involves a comprehensive data collection process to provide the traveler with essential information about their destination. It includes researching and compiling details on various accommodations, ranging from budget-friendly hostels to luxury hotels, as well as estimating the cost of living in the area. The task also covers transportation options, visa requirements, and any travel advisories that may be relevant.
        consider also the weather conditions forcast on the travel dates. and all the events that may be relevant to the traveler during the trip period.
        
        Traveling from : {from_city}
        Destination city : {destination_city}
        Arrival Date : {date_from}
        Departure Date : {date_to}

        Follow this rules : 
        1. if the {destination_city} is in a French country : Respond in FRENCH.
        """,
        expected_output=f"""
        if the {destination_city} is in a French country : Respond in FRENCH.
        In markdown format : A detailed markdown report that includes a curated list of recommended places to stay, a breakdown of daily living expenses, and practical travel tips to ensure a smooth journey.
        """,
        agent=agent,
        output_file='city_report.md',
    )
# Task: Location
def guide_task(agent, destination_city, interests, date_from, date_to):    
    return Task(
        description=f"""
        if the {destination_city} is in a French country : Respond in FRENCH.
        Tailored to the traveler's personal {interests}, this task focuses on creating an engaging and informative guide to the city's attractions. It involves identifying cultural landmarks, historical spots, entertainment venues, dining experiences, and outdoor activities that align with the user's preferences such {interests}. The guide also highlights seasonal events and festivals that might be of interest during the traveler's visit.
        Destination city : {destination_city}
        interests : {interests}
        Arrival Date : {date_from}
        Departure Date : {date_to}

        Follow this rules : 
        1. if the {destination_city} is in a French country : Respond in FRENCH.
        """,
        expected_output=f"""
        An interactive markdown report that presents a personalized itinerary of activities and attractions, complete with descriptions, locations, and any necessary reservations or tickets.
        """,

        agent=agent,
        output_file='guide_report.md',
    )
# Task: Planner
def planner_task(context, agent, destination_city, interests, date_from, date_to):
    return Task(
        description=f"""
        This task synthesizes all collected information into a detaileds introduction to the city (description of city and presentation, in 3 pragraphes) cohesive and practical travel plan. and takes into account the traveler's schedule, preferences, and budget to draft a day-by-day itinerary. The planner also provides insights into the city's layout and transportation system to facilitate easy navigation.
        Destination city : {destination_city}
        interests : {interests}
        Arrival Date : {date_from}
        Departure Date : {date_to}

        Follow this rules : 
        1. if the {destination_city} is in a French country : Respond in FRENCH.
        """,
        expected_output="""
        if the {destination_city} is in a French country : Respond in FRENCH.
        A rich markdown document with emojis on each title and subtitle, that :
        In markdown format : 
        # Welcome to {destination_city} :
        A 4 paragraphes markdown formated including :
        - a curated articles of presentation of the city, 
        - a breakdown of daily living expenses, and spots to visit.
        # Here's your Travel Plan to {destination_city} :
        Outlines a daily detailed travel plan list with time allocations and details for each activity, along with an overview of the city's highlights based on the guide's recommendations
        """,
        #context=[location_task, guide_task],
        context=context,
        agent=agent,
        output_file='travel_plan.md',
    )

location_task = location_task(
  location_expert,
  from_city,
  destination_city,
  date_from,
  date_to
)

guide_task = guide_task(
  guide_expert,
  destination_city,
  interests,
  date_from,
  date_to
)

planner_task = planner_task(
  [location_task, guide_task],
  planner_expert,
  destination_city,
  interests,
  date_from,
  date_to,
)

from crewai import Crew, Process
crew = Crew(
    agents=[location_expert, guide_expert, planner_expert],
    tasks=[location_task, guide_task, planner_task],
    process=Process.sequential,
    full_output=True,
    share_crew=False,
    verbose=True
)
result = crew.kickoff()
