import os
from datetime import datetime, timedelta
import pandas as pd

from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.llms import OpenAI
import streamlit as st
from streamlit.logger import get_logger

from moexalgo import Market

logger = get_logger(__name__)

TOKEN = st.secrets["TOKEN"]


# # Page title
st.set_page_config(page_title='🛋️👨‍💻 GPT для анализа данных Московской биржи')
st.title('🛋️👨‍💻 GPT для анализа данных Московской биржи')

st.text("""MVP решения от команды "Диванные эксперты" хакатона Go Algo от Московской биржи""")


def _generate_date_range(start_date, end_date):
    date_range = [start_date + timedelta(days=x) for x in range((end_date - start_date).days)]
    return date_range


@st.cache_data
def load_data(option, start_date, end_date):
    stocks = Market('stocks')
    dates = _generate_date_range(start_date, end_date)

    result_df = pd.DataFrame()

    if option == 'tradestats':
       for date in dates:
           response = stocks.tradestats(date=date)
           df = pd.DataFrame(response)
           result_df = pd.concat([result_df, df], ignore_index=True)
 
       with st.expander('Предпросмотр полученных данных по сделкам:'):
         st.write(result_df.head(10))
       
       return result_df

    if option == 'orderstats':   
       for date in dates:
           response = stocks.orderstats(date=date)
           df = pd.DataFrame(response)
           result_df = pd.concat([result_df, df], ignore_index=True)
 
       with st.expander('Предпросмотр полученных данных по заявкам:'):
         st.write(result_df.head(10))
       
       return result_df
    
    if option == 'obstats':
       for date in dates:
           response = stocks.obstats(date=date)
           df = pd.DataFrame(response)
           result_df = pd.concat([result_df, df], ignore_index=True)
       
       with st.expander('Предпросмотр полученных данных по стаканам заявок:'):
         st.write(result_df.head(10))
   
       return result_df


# Generate LLM response
def generate_response(df, input_query):
  llm = ChatOpenAI(model_name='gpt-3.5-turbo-0613', temperature=0.2, openai_api_key=TOKEN)
  agent = create_pandas_dataframe_agent(llm, df, verbose=True, agent_type=AgentType.OPENAI_FUNCTIONS)
  response = agent.run(input_query)
  return st.success(response, icon="✅")


# Streamlit app
def main(): 
    # Options menu with business mapping
    options_mapping = {
        "Сделки": "tradestats",
        "Заявки": "orderstats",
        "Стакан заявок": "obstats",
    }

    options = ["Сделки", "Заявки", "Стакан заявок"]

    # Date selection - Start Date
    start_date = st.date_input("Выберите начало периода:", datetime.today(), key="start_date")

    # Date selection - End Date
    end_date = st.date_input("Выберите конец периода:", datetime.today(), key="end_date")

    # Options selection
    selected_option = st.radio("Выберите нужные данные для анализа:", options=list(options_mapping.keys()), index=None)

    question_list = [
      'Какая акция самая дорогая?',
      'По какой акции было больше всего сделок?',
      'Какая была средняя разница между ценой начала и ценой окончания торгов?',
      'Другое',
    ]
    query_text = st.selectbox('Выберите пример вопроса:', question_list, disabled=not selected_option)

    if query_text == 'Другое':
      query_text = st.text_input('Введите ваш запрос:', disabled=not selected_option)
    if selected_option is not None:
      st.header('Результаты:')
      try:
        result = load_data(options_mapping[selected_option], start_date, end_date)
        generate_response(result, query_text)
      except Exception as e:
        st.error(f"""
                 Проблемы с обработкой {selected_option} - что-то сервисом (а точнее {str(e)}) 😲
                 Повторите попытку позднее ❤️
                 """)


if __name__ == "__main__":
    main()
