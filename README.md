Millions of players across the world create hundreds of thousands of DnD characters every day, give them a race, a profession, a body type, mental and emotional characteristics and a name, and label them 'good' or 'evil'. We saw these characters as a unique window into human opinion on good and evil and set out to see what the data could tell us on what our societies deem as good and what they deem evil.

Our intuition told us that race and class (profession) would be the biggest indicators on the good/evil scale and our intial data exploration indicated that beyond that, good and evil characters might not have the same attribute distributions. Good characters seem to be stronger than evil characters. Evil characters more charismatic. How significant are these differences and what do they mean?

We set out to answer these questions and to answer a larger question: given a character - it's race, background and characteristics, could we predict what kind of character the player was creating, could we predict if it was created to be good or evil.

The dataset comes from [this git repository](https://github.com/oganm/dnddata/tree/master). The repo contains several datasets collected from the owner's web apps for creating and updating DnD character sheets, and represents approximately 8000 user submitted DnD characters over the course of four years: 2018-2022.

Each Key Performance Indicator (KPI) serves as evidence of a trend or concept discernible within the data. Our aim is to validate the accuracy of our predictions by scrutinizing these instances, ensuring our model has comprehensively understood the data rather than merely making educated guesses.

# What are the different files are?
Please ensure you're using a virtual environment (venv) to prevent compatibility issues, especially with certain versions of pandas.

Execute main.py to generate a textual log detailing each phase of our workflow. Initially, we standardize and prepare the data for model training. Subsequently, we proceed with predictions, followed by an evaluation phase where we assess the results using Key Performance Indicators (KPIs).

For a comprehensive overview, refer to the accompanying PDF document, which includes detailed graphs and the complete notebook. Additionally, review the provided CSV file to gain insight into the dataset.