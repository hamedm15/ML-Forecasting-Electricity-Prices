# Log Book

## Overview

This directory served as my log book for the project, providing a detailed and organized record of the progress, decisions, and insights I gathered throughout the project's lifecycle.


## Gantt Chart and Progress
Visually track the project timeline, outlining planned activities and progress.
- [Progress Log](LogBook.md)
- [Initial Gantt Chart](InitialGanttChart.pdf)
- [New Year Gantt Chart](NewYearGanttChart.pdf)

---

## Structure and Management

This log book was a crucial tool for maintaining an organized and effective record of my project's progress and decisions. It was essential for both current understanding and future reference. I followed the established format for consistency and clarity: 

- **Regular Updates**: I ensured that notes were updated regularly, ideally after every significant project activity or meeting.
- **Gantt Chart Maintenance**: The Gantt chart was regularly updated to reflect the current progress and any changes in the project timeline.
- **Documenting Decisions**: Documentation of key decisions and rationale.
- **Meeting Summaries**: Summaries of meetings with dates, attendees, key discussion points, and action items.
- **Research Insights**: Record any significant research findings or changes in project direction.
- **Reflective Learning**: I used the log book as a tool for reflection, noting not only what was done but also what was learned.
- **Cross-Referencing**: Where possible, I cross-referenced between notes, Gantt chart entries, and additional resources for context and clarity.

### Scope/Background:

This project aims to apply machine learning models to predict electricity market prices and/or balancing costs in GB. The goal is to explore how fast ML models become obsolete for the GB market without further training. To illustrate how rapidly the electricity sector is transforming, ML models could be trained for an electricity market that transforms at a slower pace or for a completely different product. Contrasting the results, this project will provide useful insights on the interplay of the pace of power system transformation and the application of ML methods. AI-backed trading and price forecasting in energy markets is attracting growing interest, but some claim that the rapid transformation of power systems challenges the application of machine learning. In other occasions, researchers have taken advantage of the ability of ML models to predict Business-As-Usual behavior and they have used discrepancies between ML predictions with actuals to identify structural changes in electricity markets. 

---
### **Date:** November 3, 2023

- **Grading Schema:**
  - **~60% Grade:** Achieved by replicating and reproducing results from existing literature.
  - **~70% Grade:** Requires taking initiative to add modifications and produce different results.
  - **~80% Grade:** Attainable through a deeper understanding of the problem, introducing novel concepts, and potentially implementing optimization/constraints.

**Professor's Advice for Literature Review**
Keep track of the following elements in the literature:
  - **Features Used as Inputs:** Document the various features used in existing models.
  - **Techniques and Rationale:** Note the techniques applied in the literature and the reasons behind their selection.
  - **Performance Metrics:** Record the performance metrics reported in the studies.
  - **Datasets:** Identify the datasets utilized for testing the methods.

Agreed on Bi-weekly meetings to ensure relvant and regular research thorughout the term.

**Deliverable:** 1-page report including action points, summary of progress, and identification of next steps.

To-Do List
1. **Identify Project Goals:** Clarify and define the specific objectives of the project.
2. **Research/Review Literature:** Conduct a thorough review of relevant studies and publications.
3. **Create a Gantt Chart:** Develop a project timeline for effective management and tracking.


---
## Project Goals

**Aim:** Apply Machine Learning Models to Predict Electricity Market Prices in GB

- **Literature Review**
- **Technical/Software Skills Development**
- **Plan and Prepare for Deliverables**
- **Explore Choices for Model Design**
- **Project Specification and Background Write up**

Software and Evaluation:
- **Feature Selection for Model Inputs**
  - Gather time series data for the GB electricity market, including:
    - Weather conditions: Temperature, wind speed, solar irradiance.
    - Datasets from National Grid, renewable energy sources: Historical prices, demand and supply data, renewable energy output.

- **Data Preprocessing**
  - Cleaning and normalization of data.
  - Splitting data into training, testing, and validation sets.

- **Model Development and Training**
  - Selecting and developing a model to solve a time series problem.
  - Exploring different approaches:
    - LSTM RNN/Supervised Learning approach.
    - CNNs for pattern recognition in time series data.
    - Dense layer for output prediction.
  - Selecting an appropriate optimizer.
  - Conducting hyper-parameter tuning.

- **Model Performance Evaluation**
  - Establishing benchmarks for accuracy and efficiency.
    - Metrics: Mean Absolute Error (MAE), Root Mean Squared Error (RMSE).
  - Assessing computational cost.

- **Analysis and Reporting**
  - Performance of the model under different market conditions.
  - Insights on the adaptability and effectiveness of ML models in rapidly changing electricity markets.
    - Discussion of methods used, adaptability, effectiveness, and limitations.

---

### Table 1: Project Milestones

| Event                     | Date & Time                 | Term              |
| ------------------------- | --------------------------- | ----------------- |
| Staff Project lists       | Monday, 9/10/2023           | Autumn            |
| Project selection         | Thursday, 26/10/2023, 14:00 | Autumn            |
| Projects allocated        | Week 5, Autumn              | Autumn            |
| Interim Report            | Monday, 12/2/2024, 16:00    | Spring            |
| Abstract and draft report | Monday, 27/5/2024, 16:00    | Summer            |
| Final Report              | Friday, 14/6/2024, 16:00    | Summer            |
| Presentations             | 24-27/6/2024                | Last week, Summer |

## Criteria and Deliverables

### Table 2: Project Deliverables

| Deliverable             | Weighting | Purpose                                                           |
| ----------------------- | --------- | ----------------------------------------------------------------- |
| Interim Report          | 10%       | Initial research and early results, with a detailed project plan  |
| Abstract & Title Update | 0%        | Accompanying the final report                                     |
| Final Report            | 77.5%     | Comprehensive documentation of project findings and methodologies |
| Presentation            | 12.5%     | Showcase of achievements, Q&A session                             |

### Criteria for Success

1. **Independent Problem-Solving**:

   - Work independently in defining and solving technical problems.
   - Supervisory support is provided for guidance, and to identify aspects of the work that fulfill this criterion.
   - Starting technical work early in the project is crucial to mitigate risks and inform the Interim Report effectively.
2. **Background Research**:

   - Locating and researching additional project background material.
   - Balance the depth of research, ensuring time is not spent excessively on irrelevant background.
   - Regular check-ins with the supervisor to confirm the relevance and sufficiency of the research material.
3. **Critical Review of Background Material**:

   - Write a critical review of an appropriate part of the background material.
   - This involves understanding how the material relates to the project problem and articulating this understanding effectively.
   - The review will likely contribute to the Final Report, potentially after some revisions.
4. **Self-propelled Work Ethic**:

   - take control of all organization aspects necessary for efficient project work.
   - Proactive communication with the supervisor about progress and challenges, arranging regular meetings, setting targets for future work, and detailed planning of the project.
5. **Understanding of Expected Project Deliverables**:

   - Detail the project requirements and formulate achievable deliverables.
   - Understand the work required to achieve these deliverables and the associated uncertainties.
6. **Appropriate Use of External Help**:

   - Seeking help from others.
   -  Supervisor is the primary source of technical advice and can provide feedback on understanding of deliverables and other project aspects.
   - Seek assistance from PhD students, and other knowledgeable individuals as needed.
   - Efficient work includes identifying who can help with specific questions and actively seeking their assistance.


---

### November 2023

**Goals:**
- N/A

**Progress:**
- Project initiation and kickoff.
- Initial meeting with supervisor, Elina Spyrou.
- Establishment of project goals and timelines.

**Notes/Thoughts:**
- Focused on understanding the project's scope and requirements.
- Began initial literature review and research into electrity markets.

---
### December 2023

**Goals:**
- Begin data collection
- Make a plan/Gannt Chart
- Outline for Interim Report
- Explore choices behind selected input feautres and techniques
- Familiarise with Neural Networks in python

**Progress:**
- Continued literature review, read about 20 papers so far.
- Prepared a plan/Gannt Chart 
- Project Specifiation section of Interim Report drafted.
- Developed initial project outlines and hypotheses.
- Worked through "Intro to Statistical Analysis in Python" ML Labs.

**Notes/Thoughts:**
- Challenges encountered in identifying relevant data sources.
- Considering different ML models and their applicability/ still unsure of implementation techniques.
- Personal Deadlines set
- Spreadsheet for data collection made and github repository made.

---
### **Date:** January 10, 2024

**Attendees:** Hamed Mohammed, Dr. Elina Spyrou
#### Agenda

- Data for the GB market
- Access to relevant hardware resources (16 core): Specicially Linux based systems with CPU and GPUs to run models and evaluation, This is neccary otherwise it will not allow for ensemble/quantile averaging or advanced computaion of deep neural networks. Even with access to hardware it is estimated to take 10 hours per QRA model training. For this reason remote access would be prefered. 

To-Do List
- Collect and Compile data from given sources
- Focus on UK specific literature
- Continue with Interim Report
---

### January 2024

**Goals:**

- Submission of the Interim Report Draft.
- Complete data collection and preliminary analysis.
- Resume regular meetings with the supervisor for guidance.

**Progress:**
- Continued literature review, drafted Statistical Methods section of the report.
- Began Data collection for UK markets.
- Identified specific imnput features. 


**Notes/Thoughts:**
- Reconsidering Project Goals and Specification.

---
### February 2024

**Goals:**
- Submission of the Interim Report.
**Notes/Thoughts:**
- Data preprocessing and model training commenced.
- Experimenting with different ML algorithms.
- First set of results analyzed.
- Encountering some overfitting issues with initial models.
- Exploring parameter tuning and model optimization strategies.

---

### April 2024
**Notes/Thoughts:**
- Continued model refinement and testing.
- Began drafting sections of the Final Report.
- Attended relevant workshops and seminars for additional insights.
- Finding a balance between model complexity and performance.
- Gaining deeper understanding of the energy market dynamics.

---

### May 2024
**Notes/Thoughts:**
- Further model optimization.
- Started compiling results and findings for the Final Report.
- Began preparation for the presentation.
- Time management for report writing and presentation preparation.
- Submission of the abstract and draft report.
- Ensuring all findings and methodologies are well-documented.

---

### June 2024
**Notes/Thoughts:**
- Contemplating future work and applications of the research.
- Finalization of the Final Report.
- Submission of the Final Report.
- Rehearsals for the final presentation.
- Conducting the final presentation.

