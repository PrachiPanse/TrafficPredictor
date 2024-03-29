from common.types import Whole_Options, Particular_Options
import streamlit as st
import pandas as pd

def particular_operations(df, question, time, location):
    df['least_traffic_hour']= df['prediction'].apply(lambda x: x.argmin()+1)
    df['most_traffic_hour'] = df['prediction'].apply(lambda x: x.argmax()+1)

    chart_data = pd.DataFrame({
        "Time": range(1,len(df.loc[0,'prediction'])+1),
        "Pedestrian Count": df.loc[location, 'prediction'],
    })
   
    match question:
        case Particular_Options.least_traffic_hour:
            answer = df.loc[location, 'least_traffic_hour']  
            st.write(f"Traffic at location #{location+1} is expected to be least at {answer}h from now.")

        case Particular_Options.most_traffic_hour:
            answer =  df.loc[location, 'most_traffic_hour']
            st.write(f"Traffic at location #{location+1} is expected to be highest at {answer}h from now.")
        
        case Particular_Options.num_pedestrians:
            answer = df.loc[location, 'prediction'][time-1]
            st.write(f"The number of pedestrians at location #{location+1} at time {time}h from now is expected to be {answer}.")
        
        case Particular_Options.prediction_24h:
            answer = df.loc[location, 'prediction']
            st.write(f"The traffic prediction at location #{location+1} over the next 24h is indicated below:")
            st.dataframe(
                data = chart_data.transpose(),
                use_container_width=True, 
                width = 600,
                hide_index= False
            ) 


            st.bar_chart(
                chart_data, 
                x="Time", 
                y="Pedestrian Count", 
                color=["#5000FF"]
            )

    return None

####################################################################################


def whole_operations(df, question, time):  
    match question:
        case Whole_Options.least_traffic_areas:#at a particular time
            answer = df['prediction'].apply(lambda x: x[time-1]).argmin() + 1
            st.write(f"At time {time}h from now, the least traffic is expected to be at location #{answer}.") 

        case Whole_Options.most_traffic_areas:#at a particular time
            answer = df['prediction'].apply(lambda x: x[time-1]).argmax() + 1
            st.write(f"At time {time}h from now, the highest traffic is expected to be at location #{answer}.")
        
        case Whole_Options.avg:  #average over all locations
            sum = df['prediction'].apply(lambda x: x[time-1]).sum()
            answer = int(sum/len(df['prediction']))
            st.write(f"At time {time}h from now, the average traffic at any location in the city is expected to be {answer}.")
        
        case Whole_Options.num_pedestrians: #at a particular time
            answer = df['prediction'].apply(lambda x: x[time-1])
            st.write(f"The number of pedestrians at a time {time}h from now, for various locations in the city is indicated below: ")
            
            location_wise_pedestrian = []
            location_wise_pedestrian = [df.loc[x, 'prediction'][time-1] for x in range(len(df))]

            chart_data = pd.DataFrame({
                "Location": range(1,len(df)+1),
                "Pedestrian Count": location_wise_pedestrian,  
            })

            st.dataframe(
                data = chart_data.transpose(),
                use_container_width=False, 
                width = 400,
                hide_index= False
            ) 

            st.bar_chart(
                chart_data, 
                x="Location", 
                y="Pedestrian Count", 
                color=["#5000FF"]
            )

    return None



