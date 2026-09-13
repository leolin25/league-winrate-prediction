# league-winrate-prediction
League of legends (LoL) is a MOBA created by Riot Games and highly competitive, the average league game can last about 25-35 mins but most of the time in ranked matches lots of variable can happen early on which leads to a massive advantage for a team early on. The aim of this project is to analyse high-diamond elo games in league of legends at 10 mins and determine the chance of winning.

Data:
- Obtained from Kaggle: https://www.kaggle.com/datasets/bobbyscience/league-of-legends-diamond-ranked-games-10-min/data?select=high_diamond_ranked_10min.csv
- Data is about DIAMOND-MASTER ranked matches before 10 mins

Glossary/Key terminology (from Kaggle)
- Warding totem: An item that a player can put on the map to reveal the nearby area. Very useful for map/objectives control.
- Minions: NPC that belong to both teams. They give gold when killed by players.
- Jungle minions: NPC that belong to NO TEAM. They give gold and buffs when killed by players.
- Elite monsters: Monsters with high hp/damage that give a massive bonus (gold/XP/stats) when killed by a team.
- Dragons: Elite monster which gives team bonus when killed. The 4th dragon killed by a team gives a massive stats bonus. The 5th dragon (Elder Dragon) offers a huge advantage to the team.
- Herald: Elite monster which gives stats bonus when killed by the player. It helps to push a lane and destroys structures.
- Towers: Structures you have to destroy to reach the enemy Nexus. They give gold.
- Level: Champion level. Start at 1. Max is 18 (when data was posted on Kaggle)


Files:
- eda.ipynb: This file documents the eda process of the project, I used this file to visualise intresting data and also document discoveries I made about the data
- features.py: Python file for feature engineering
- model.py: Python file containing the model for the project