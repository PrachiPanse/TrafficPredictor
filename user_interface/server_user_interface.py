import streamlit as st
from user_interface.prediction_handler import particular_operations, whole_operations
from common.types import Whole_Options, Particular_Options, Question_Types

def create_user_interface(df):
    #Title
    st.title(":woman-walking: Pedestrian Traffic Predictor :walking:")
   
    #Side message
    with st.sidebar:
        message = '<p style="font-family:comic-sans; color:gray; font-size: 20px;">General Information:</p>'
        st.markdown(message, unsafe_allow_html=True)

        message = '<p style="font-family:comic-sans; color:gray; font-size: 16px;">This tool answers user queries about the pedestrian traffic in the city. It utilizes a Neural Network model to make forecasts for up to 24 hours of the hourly traffic at various locations in the city.</p>'
        st.markdown(message, unsafe_allow_html=True)

        message = '<p style="font-family:comic-sans; color:gray; font-size: 16px;">\nPlease select from the options according to your requirement.</p>'
        st.markdown(message, unsafe_allow_html=True)
        
    message = '<p style="font-family:comic-sans; color:Black; font-size: 25px;">Hello!! I am a Pedestrian Traffic Predictor</p>'
    st.markdown(message, unsafe_allow_html=True)

    #Question
    question_type = st.radio(
        label = "Would you like to know about the traffic across the city or at a particular location?",
        options = Question_Types,
        horizontal = True,
        index=None,
        format_func=lambda x: x.value,
    )
    
####################################################################################
    #Forecasting for the city overall
    if question_type == Question_Types.whole:
        st.subheader("Forecasting across the city", divider='blue')
        
        question_whole = st.selectbox(
            label = f"What would you like to know about the traffic condition in the city?", 
            options = Whole_Options, 
            index=None, 
            format_func=lambda x: x.value, 
            placeholder="Choose an option", 
        )
        
        if question_whole:  
            desired_time = st.selectbox(
                label = f"Specify the time offset from now you are interested in", 
                options = list(range(1, len(df.loc[0,'prediction'])+1)),
                placeholder="Choose an option",
                index=None,
            )
            if desired_time:
                whole_operations(df, question_whole, desired_time)   

####################################################################################
    #Forecasting for a chosen location in the city
    elif question_type == Question_Types.particular:
        st.subheader("Forecasting for a chosen location in the city", divider='blue')
        location = st.selectbox(   #TODO make time T1 T2, etc to resolve confusion
            label = f"Specify the location you are interested in",
            options = list(range(1, len(df['prediction'])+1)),
            placeholder="Choose an option",
            index=None,
        )  
        if location:     
            question_particular = st.selectbox(
                label = f"What would you like to know about the traffic condition at this location?",
                options = Particular_Options,
                format_func = lambda x: x.value,
                placeholder="Choose an option",
                index=None,
            )
            if question_particular:
                if question_particular == Particular_Options.num_pedestrians:
                    desired_time = st.selectbox(
                        label = f"Specify the time you are interested in:",
                        options = list(range(1, len(df.loc[0,'prediction'])+1)),
                        placeholder="Choose an option",
                        index=None,
                    )
                    if desired_time:
                        particular_operations(df, question_particular, desired_time, location-1)
                else:
                    particular_operations(df, question_particular, 0, location-1)


    return None



if __name__ == "__main__":
    create_user_interface({})
