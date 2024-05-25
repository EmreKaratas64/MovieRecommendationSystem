# Movie Recommender System Experiment

This repository contains a Python script and necessary datasets to reproduce an experiment for a movie recommender system that uses both rule-based and clustering-based approaches.

## Prerequisites
- Python 3.8 or higher
- Libraries: pandas, numpy, scikit-learn

## Installation
First, ensure that Python and pip are installed on your system. You can install the required Python libraries using:

```bash
pip install pandas numpy scikit-learn
```

## How to Reproduce the Experiment

### Step 1: Clone the Repository
Clone this repository to your local machine by running:
```bash
git clone https://github.com/EmreKaratas64/MovieRecommendationSystem.git
```

### Step 2: Navigate to the directory
Open your terminal and navigate to the directory where the repository is cloned.

### Step 3: Running the Script
Reproduce the experiment by running:
```bash
python Movie_Recommendation.py
```

### The Output
The script will perform the following operations:
1. Load the `ratings.csv` dataset.
2. Preprocess and normalize the data.
3. Split the data into training and test sets.
4. Implement rule-based and clustering-based recommendation systems.
5. Combine the outputs of both recommenders.
6. Evaluate the system using the Mean Squared Error (MSE) and denormalized MSE.

The final output will display the MSE and the denormalized MSE for the combined recommender system.

## Expected Results
You should see the MSE and denormalized MSE printed in your terminal after you start the program. This will indicate the performance of the recommender system.

## File Descriptions
- `Movie_Recommendation.py`: Main Python script for the experiment.
- `movies.csv`: Dataset containing movie details such as movie ID, title, and genres.
- `ratings.csv`: Dataset containing user rating details such as user ID, movie ID, rating and timestamp.
- `results.png`: PNG file which shows the produced results after running the `Movie_Recommendation.py` correctly.
- `ProgramNotes.txt`: Tex file containing information regarding runtime which states "The Program runtime is about 2-3 minutes!".

For any issues or further queries, refer to the comments in the `Movie_Recommendation.py` script for more detailed information about the functions and methodology used.
