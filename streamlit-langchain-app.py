import os
from datetime import datetime, timedelta, date
import pandas as pd
import numpy as np

from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.llms import OpenAI
import streamlit as st
from dotenv import load_dotenv

from moexalgo import Market, Ticker


# # Page title
st.set_page_config(page_title='🦜🔗 GPT для анализа данных биржи')
st.title('🦜🔗 GPT для анализа данных биржи')

load_dotenv()

TOKEN = os.getenv("TOKEN")


def _generate_date_range(start_date, end_date):
    date_range = []
    current_date = start_date

    while current_date <= end_date:
        date_range.append(current_date.date())
        current_date += timedelta(days=1)

    return date_range


# Define your business logic functions or variables
def tradestats(start_date, end_date):
    stocks = Market('stocks')
    dates = _generate_date_range(start_date, end_date)

    result_df = pd.DataFrame()

    for date in dates:
        df = stocks.tradestats(date=date)
        result_df = pd.concat([result_df, df], ignore_index=True)
        print(f'date {date} loaded with {len(df.index)} rows')

    with st.expander('See DataFrame'):
      st.write(result_df)
    
    return result_df


def orderstats(start_date, end_date):
    stocks = Market('stocks')
    dates = _generate_date_range(start_date, end_date)

    result_df = pd.DataFrame()

    for date in dates:
        df = stocks.orderstats(date=date)
        result_df = pd.concat([result_df, df], ignore_index=True)
        print(f'date {date} loaded with {len(df.index)} rows')

    with st.expander('See DataFrame'):
      st.write(result_df)
    
    return result_df


def obstats(start_date, end_date):
    stocks = Market('stocks')
    dates = _generate_date_range(start_date, end_date)

    result_df = pd.DataFrame()

    for date in dates:
        df = stocks.obstats(date=date)
        result_df = pd.concat([result_df, df], ignore_index=True)
        print(f'date {date} loaded with {len(df.index)} rows')
    
    with st.expander('See DataFrame'):
      st.write(result_df)

    return result_df



# Generate LLM response
def generate_response(df, input_query, openai_api_key):
  llm = ChatOpenAI(model_name='gpt-3.5-turbo-0613', temperature=0.2, openai_api_key=openai_api_key)
  # Create Pandas DataFrame Agent
  agent = create_pandas_dataframe_agent(llm, df, verbose=True, agent_type=AgentType.OPENAI_FUNCTIONS)
  # Perform Query using the Agent
  response = agent.run(input_query)
  return st.success(response, icon="✅")


# Streamlit app
def main():
    # Options menu with business mapping
    options_mapping = {
        "Сделки": tradestats,
        "Заявки": orderstats,
        "Стакан заявок": obstats,
    }

    # Date selection - Start Date
    start_date = st.date_input("Выберите начало периода:", datetime.today(), key="start_date")

    # Date selection - End Date
    end_date = st.date_input("Выберите конец периода:", datetime.today(), key="end_date")

    # Options selection
    selected_option = st.radio("Выберите нужные данные для анализа:", options_mapping.keys())

    question_list = [
      'Какая акция самая дорогая?',
      'По какой акции было больше всего сделок?',
      'Какая акция имеет большие шансы на рост цены?',
      'Другое']
    query_text = st.selectbox('Выберите пример вопроса:', question_list, disabled=not selected_option)
    openai_api_key = st.text_input('Введите OpenAI API Key', type='password', value=TOKEN, disabled=not (selected_option and query_text))

    if query_text == 'Другое':
      query_text = st.text_input('Введите ваш запрос:', disabled=not selected_option)
    if not openai_api_key.startswith('sk-'):
      st.warning('Введите ваш OpenAI API key!', icon='⚠')
    if openai_api_key.startswith('sk-') and (selected_option is not None):
      st.header('Результаты:')
      try:
        result = options_mapping[selected_option](start_date, end_date)
        generate_response(result, query_text, openai_api_key)
      except Exception as e:
        st.error(f"Проблемы с обработкой {selected_option} - что-то сервисом (а точнее {str(e)}). Повторите попытку позднее")

    


if __name__ == "__main__":
    main()
