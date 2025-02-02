# air-traffic-control-simulator

#### This repository contains the source code for our air traffic control simulator. It takes in as input trajectory data of flights over a fixed time series and outputs a series of air traffic commands for each individual aircraft in order to ensure safety while minimising time delay and fuel consumption. 

#### Inspiration

With the recent rise in aerial collisions caused by inefficient air traffic management, and a growing shortage of air traffic controllers struggling to keep up with increasing global air traffic, the need for innovative solutions has never been greater. ATCAssist leverages AI to develop a deep understanding of complex decision-making logic, providing intelligent suggestions for air traffic commands. By easing the workload of ATC staff, ATCAssist enhances safety, efficiency, and overall airspace management.

#### What it does

It sends the current game state into a trained XGBoost decision tree, which will run the inference and generate the next predicted state of all flights.

#### How we built it

To train our XGBoost decision tree model, we began by extracting ATC commands and preprocessed them to align with flight trajectory data. We then ran inference on the model using input game state data, which was filtered using a low-pass filter to ensure accuracy. For each flight, we generated a time series of suggested commands over the simulation environment's timeline. These commands were subsequently fed into a data visualization tool, allowing ATC staff to easily visualize and interpret the recommended flight paths.

#### Challenges we ran into

We initially wanted to write an optimisation model which would determine the next game state based on an established set of constraints and cost function. However, mapping competing objectives like reaching the target airport and not entering the crash threshold zone of another aircraft into our model was extremely challenging and thus made it infeasible. As such, many hours into the Hackathon, we decided to pivot to a machine learning model which would generate the next "optimal" game state based on historical data. This, though might not definitely conform to literature review of air traffic control principles, turned out to be accurate to real-life decisions made by ATC staff.

#### Accomplishments that we're proud of

Learning optimisation and its libraries from scratch, and pushing through and bouncing back with a backup plan amidst all the difficulties.

#### What we learned

Optimisation is greatly limited.

#### What's next for ATCAssist

Integrating a performance evaluation metric for constant retraining loop for improved performance!
