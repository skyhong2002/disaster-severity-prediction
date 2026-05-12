Data Mining | Spring 2026 Final Project

> please refer to the pdf to get full description which is the original slides

## Data Mining

Spring 2026

# Final Project:

## Natural Disaster Severity Prediction

## April 30, 2026


## Outline

#### ● Introduction

#### ● Dataset Overview

#### ● Objective

#### ● Rules

#### ● Submission & Grading Criteria

#### ● Reference


## Introduction

#### The final project will be conducted as a Kaggle Competition.

###### ● It is an Group Work and counts for 35% of the semester grade

###### ● Final Project Kaggle Link

#### ● Please sign in with your NYCU email account directly.

###### ● Deadline: June 10, 11:55 pm

###### ● Core Task:

###### ○ Predict natural disaster severity levels for the next five weeks using only

###### historical meteorological data.


## Introduction

###### Invite team members

#### Team -> Send invitation -> Click continue

```
search user name (not display name)
user name
```

## Kaggle Submission Rule

```
● Name limitation :
Change the team name in Kaggle as Team {Group ID} (e.g. Team 1, Team 21).
● Daily Limit :
Maximum of 3 submissions per team per day (reset on UTC+0, i.e. 8 AM in Taiwan).
● Evaluation : Submissions are evaluated using MAE (Mean Absolute Error).
● The private leaderboard will only be visible after the competition ends.
○ By default, your two best public leaderboard submissions will be used for the private
leaderboard.
```

## Dataset

We are using a Natural Disaster Severity Prediction Dataset:
● **Files** : Includes train.csv and test.csv.
● **Columns** :
○ region_id: Unique identifiers for different
geographic regions.
○ date: Observation dates.
■ In “train.csv”, each region has a total of 5,
days.

### ○ Other Features: meteorological data


## Dataset

7
days
7
days

###### ● Label (the column “score”):

###### ○ Ranges from 0 to 5, with higher values indicating greater severity of the disaster.

###### ■ Natural disaster type: Drought

###### ○ Scores are recorded weekly

###### ■ The dataset contains NaN values for the other 6 days of the week.

###### ■ The score values in the provided datasets are all integers. Your predictions can

###### be either integers or floats.


## Objective

#### ● Task:

###### Train a model based on “train.csv”, then use the 91 days of data for each

###### region provided in “test.csv” as input to predict each region’s disaster severity

#### levels for the next five weeks.


## Training Data Format

```
Region ID Date XXX Score
R1 XXXX-09-18 XXX NAN
.
.
.
.
.
.
.
.
.
.
.
.
.
2
.
.
R1 XXXX-12-17 XXX 1
R1 XXXX-12-18 XXX NAN
```
........
R
   .........
      .........
          1. 2. 1. 2. 1

#### Train.csv and region_id = R

#### 91 days

#### 35 days

#### Model

###### Predict the score

###### columns for

###### next 5 consecutive

###### weeks

###### (after 91 days)


## Testing Data Format

```
Region ID Date XXX
R1 XXXX-09-18 XXX
.
.
.
.
.
.
.
.
.
.
.
.
R1 XXXX-12-17 XXX
R2 XXXX XXX
.
.
.
.
.
.
.
.
.
```
#### Test.csv

#### 91 days

#### 91 days

#### Model

###### Predict the score

###### columns for

###### next 5 consecutive

###### weeks

###### (after 91 days)

no “score” column


## Kaggle Submission Format

#### Inference

#### Model [1.5, 2.0, 2.1, 0.2, 1.2]^

#### R1_data

#### in test.csv

#### submission.csv

#### R1_future_score

#### ● Format: We provide a template submission file

#### (“sample_submission.csv”).


## Grading Criteria

#### The grade for the Final Project [35 points] is based on:

#### 1. Kaggle & Report [30 points]

#### a. Kaggle Competition (60%)

#### b. Your Report (40%)

#### 2. Progress Check (May 21) [5 points]


## Grading Criteria

#### The grade for the Final Project [35 points] is based on:

#### 1. Kaggle & Report [30 points]

#### a. Kaggle Competition (60%)

#### b. Your Report (40%)

#### 2. Progress Check (May 21) [5 points]


## Grading Criteria - Kaggle & Report [30 points]

###### ● Kaggle Competition (60%)

```
○ 30% based on Public Leaderboard Points (40% of testing data)
○ 70% based on Private Leaderboard Points (remaining 60% of testing data)
```
###### ● Three Baselines

```
Public Leaderboard
(Private Leaderboard is hidden until the end of the competition)
```

## Grading Criteria - Kaggle & Report [30 points]

###### ● Kaggle Competition (60%)

○ 30% based on Public Leaderboard Points

###### ○ 70% based on Private Leaderboard Points

###### Leaderboard Point Calculation

```
Your Model Performance Points Calculation Method
Above Baseline 3 42 ~ 60
Between Baselines 2 & 3 25 ~ 40
Between Baselines 1 & 2 10 ~ 25
Under Baseline 1 5 5
N: the number of people over
Baseline 3
(If N=1, then the point = 60)
```

## Grading Criteria

#### The grade for the Final Project [35 points] is based on:

#### 1. Kaggle & Report [30 points]

#### a. Kaggle Competition (60%)

#### b. Your Report (40%)

#### 2. Progress Check (May 21) [5 points]


## Grading Criteria - Kaggle & Report [30 points]

**- Report (40%)**
    **○ Format**
       **■** Please use the provided Overleaf template to write your report
          ● Overleaf template is built on IEEE paper template
          ● You may also use the Microsoft Word template provided by IEEE (please download
             the **A4 format** version)
       ■ Modification of the basic format is not allowed (e.g., font size, line spacing).


## Grading Criteria - Kaggle & Report [30 points]

**- Report (40%)**
    **○ Content (Total 5-8 Pages, Not Include References)**
       i. **Abstract:** Include your group ID and Github link (with README.md in Github repo)
          ● README.md: Describe in detail how to run your code.
**ii. Project Summary (Max 1 Page)**
● Describe the goal of this project
● Provide the high-level idea of your method
● Introduce the related works
**iii. Proposed Method (1-3 Pages)**
● Please describe your proposed method in detail. You may include formulas, pseudocode,
framework diagrams, etc.
**iv. Experiments (2-4 Pages)**
● Data description, statistics of data, and the basic setup for your experiments.
● You can also include your self-defined baselines to reveal the robustness of your method.
● We encourage you to include experiments and analysis from different perspectives, such as
○ Your model’s performance on Kaggle (you can compare current and past results)
○ Comparison with your self-defined baselines
○ Ablation study, Parameter analysis, Case study, etc.


## Grading Criteria

#### The grade for the Final Project [35 points] is based on:

#### 1. Kaggle & Report [30 points]

#### a. Kaggle Competition (60%)

#### b. Your Report (40%)

#### 2. Progress Check (May 21) [5 points]


## Grading Criteria - Progress Check [5 points]

**- Date:** May 21
**- Progress Check:**
    ○ **5-minute** presentation for each group
    ○ You may briefly describe your current observations and any challenges you have
       encountered.
    ○ Grading will be based on the completeness of the progress you present.
    ○ There is no need to send the slides to the TAs in advance, but please **remember to bring**
       **your own laptop** for the presentation.


## Grading Criteria

- You are allowed to use the AI tools to assist your work, but ...
    ○ Any form of plagiarism in code or report will result in a **score of 0 for this assignment**.
    ○ The submitted **report, code, and Kaggle results must be consistent and reproducible**.
       ■ If the code cannot be executed, or if the Kaggle results are inconsistent with the
          submitted code or report, this will also result in a **score of 0 for this assignment**.
    ○ If any **hallucinations** are found in the report (e.g., non-existent references), **the report will**
       **receive a score of 0.**

##### If there are concerns about your submission,

##### TAs may require an on-site demonstration to verify the results.


## Report and Code Submission

```
Please submit your report & code by June 10, 11:55 PM.
Late submissions are not accepted.
Submissions consist of two parts (Besides Kaggle Submission):
```
**1. Code**
    **a.** Please host your **code on a public GitHub repository** (commit before deadline).
2. **Report.**
    **a.** Please ensure your public **GitHub link is included** within your report, and submit your **report**
       **via E3.
b. File name:** DM_project_Group_{GroupID}.pdf (e.g., DM_project_Group_17.pdf)
Please write down your report in English.
The TA will test whether your code can run successfully; additionally, your **Kaggle Display Name**
must be **Team {Group ID}** , or your grade will be recorded as 0.
If you have any other questions, please feel free to contact the TA at nycu.dm.ta@gmail.com.


## Optional Oral Presentation

**- Date:** June 11 (Final week of the DM course)
- **Grading Criteria** [Up to 5% extra credit to the semester grade]:

###### ○ Based on the completeness of your final project and your presentation

- **First come, First served:**

###### ○ Due to limited class time, groups that sign up earlier will have priority to present.

- Sign up link and other information will be released later.


## Reference

**- Here are some papers related to other Kaggle competitions for your reference**
    **○** Iglovikov, Vladimir, Sergey Mushinskiy, and Vladimir Osin. "Satellite imagery feature detection
       using deep convolutional neural network: A kaggle competition." _arXiv preprint arXiv:1706.06169_
       (2017).
    ○ Kechyn, Glib, et al. "Sales forecasting using WaveNet within the framework of the Kaggle
       competition." _arXiv preprint arXiv:1803.04037_ (2018).


