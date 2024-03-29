from enum import Enum

#################################################################################################
#################################### Dataset mapping dictionary ###################################
#################################################################################################
class DatasetType (Enum):
    dataset_1_pedestrian = 'dataset_1_pedestrian'
    dataset_1_pedestrian_small = 'dataset_1_pedestrian_small'
    dataset_1_pedestrian_5_series = 'dataset_1_pedestrian_5_series'
    #Needed for using in argparse - https://stackoverflow.com/questions/43968006/support-for-enum-arguments-in-argparse
    def __str__(self):
        return self.value

dataset_mapping_dict = {
    DatasetType.dataset_1_pedestrian : './datasets/1_pedestrian_counts/pedestrian_counts_dataset.tsf',
    DatasetType.dataset_1_pedestrian_small : './datasets/1_pedestrian_counts/pedestrian_counts_dataset_small.tsf',
    DatasetType.dataset_1_pedestrian_5_series : './datasets/1_pedestrian_counts/pedestrian_counts_dataset_5_series.tsf'
}



class Whole_Options(Enum):
    least_traffic_areas = "Least traffic area at a particular time" 
    most_traffic_areas = "Most traffic area at a particular time"
    num_pedestrians = "Number of pedestrians at a particular time"
    avg = "Average traffic at a particular time"

class Particular_Options(Enum):
    least_traffic_hour = "Least traffic hour in the next 24h" 
    most_traffic_hour = "Most traffic hour in the next 24h"
    num_pedestrians = "Number of pedestrians at a particular hour"
    prediction_24h = "Traffic prediction over the next 24h"

class Question_Types(Enum):
    whole = "Across the city"
    particular = "A particular location in the city"