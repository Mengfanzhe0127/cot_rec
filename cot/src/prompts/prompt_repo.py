RECOMMEND_INSTRUCTION_SCORE_LABEL = """
## Instruction
You are an expert in evaluating user preference analyses for movie recommendation systems.

You will be given the following information:
1. A **complete user interaction history**, including conversation turns and any clicked movies.
2. Several **user preference analyses**, each consisting of:
   - A step-by-step reasoning process (**Analysis**) based on the interaction history
   - A structured summary (**User Preferences**) listing inferred movie preferences and preferred/disliked attributes
3. The **final accepted movie**, along with its attributes. This movir reflects the user's true preferences.

---

### Your Task:
Your goal is to **score and rank each user preference analysis** based on how well it reflects the user's true preferences, as reflected in the final accepted movie.

A high-quality analysis should satisfy the following conditions:

1. **Consistency with the accepted movie**:
   - The listed `movies` and `attributes` in the summary should be consistent with the accepted movie and its attributes.
   - Do **not favor short or vague summaries**. Preference summaries that are overly brief and miss the important attributes in the final accepted movie should be penalized.

2. **Reasoning quality**:
   - Avoid the behavior of inventing user preferences or adding preferences not supported by the interaction history.
   - The analysis and summary must be in English only. Penalize any use of Chinese or mixed languages.
   - The **format** should follow the standard: numbered reasoning steps (e.g., 1. ... 2. ...), followed by the structured preference output.

3. **Output Format Enforcement:**
   - Movies need be in the format: Movie Title (Year) without quotes or extra characters.
   - Use **IMDB-style vocabulary** for attributes (e.g., `drama`, `Adam Sandler`)
Example of a valid output:
### User Preferences:
- movies:
  - like: The Dark Knight (2008), Titanic (1997)
  - dislike: None
- attributes:
  - like: action, drama, Christopher Nolan, Leonardo DiCaprio
  - dislike: None
---

## Input Format:
### 1. User Interaction Record
A full interaction sequence, shown in turn-based format.

### 2. Multiple User Preference Analyses
Each preference analysis includes:
Preference Analysis [Id]:

### Analysis:
[A step-by-step reasoning based on the user interaction]

### User Preferences:
- movies:
  - like: list of liked movies or None
  - dislike: list of disliked movies or None
- attributes:
  - like: list of preferred genres, styles, directors, themes, stars, etc., or None
  - dislike: list of disliked genres, styles, directors, themes, stars, etc., or None

### 3. Final Accepted Movie
This is the movie ultimately accepted by the user, representing their finally true preferences.

---

## Output Format:
### Reason:
Provide your reasoning for ranking. Focus on how well the inferred preferences align with the accepted movie's attributes and the user history. Penalize missing key elements, hallucination, or formatting issues.

Ranked Preference Analyses:
Rank the analyses from best to worst. Assign each a score from 1 to 5, where:
**5 = Excellent** =  Strong match with the accepted movie and valid reasoning
**1 = Poor** = Weak alignment, hallucinated preferences, poor structure or format
Use this format:

### Ranked Preference Analyses:
1. Preference Analysis [Id], Score: [score]
2. Preference Analysis [Id], Score: [score]
...

---

**Important rules**:
- Do not use markdown symbols (*) to identify preference analysis ids and scores when listing them.
""".strip()

RECOMMEND_INSTRUCTION_SCORE_FINAL = """
## Instruction
You are an expert in evaluating user preference analyses for movie recommendation systems.
You will be given the following information:
1. A **complete user interaction history**, including conversation turns and any clicked movies.
2. Several **user preference analyses**, each consisting of:
   - A step-by-step reasoning process (**Analysis**) based on the interaction history
   - A structured summary (**User Preferences**) listing inferred movie preferences and preferred/disliked attributes

---

### Your Task:
Your goal is to **score and rank each user preference analysis** based on how well it reflects the user's true preferences, as expressed in the interaction history.

A high-quality analysis should satisfy the following conditions:

1. **Reasoning quality**:
   - The **Analysis** section should follow the **order of interaction** and clearly explain **how** user preferences were inferred from each record.
   - Avoid the behavior of inventing user preferences or adding preferences not supported by the interaction history.
   - The analysis and summary must be in English only. Penalize any use of Chinese or mixed languages.
   - The **format** should follow the standard: numbered reasoning steps (e.g., 1. ... 2. ...), followed by the structured preference output.

2. **Preference Coverage and Grounding**:
   - The **User Preferences** section should be **fully grounded in the Analysis**. 
      - If the analysis states a user liked a movie, that movie and its attributes should appear in the like list.
      - If a disliked movie/attribute is inferred, it should appear under dislike.
   - The preference summary should **maximize coverage** of all user-expressed or implied preferences from the interaction history:
      - Include all movies the user clearly liked/disliked or positively/negatively responded to.
      - Extract relevant attributes (genres, directors, themes, actors) as **explicitly and thoroughly as possible**.
   - Do **not favor short or vague summaries**. Preference summaries that are overly brief and omit important clues should be penalized.

3. **Output Format Enforcement:**
   - Movies need be in the format: Movie Title (Year) without quotes or extra characters.
   - Use **IMDB-style vocabulary** for attributes (e.g., `drama`, `Adam Sandler`)
Example of a valid output:
### User Preferences:
- movies:
  - like: The Dark Knight (2008), Titanic (1997)
  - dislike: None
- attributes:
  - like: action, drama, Christopher Nolan, Leonardo DiCaprio
  - dislike: None

4. **Penalize Analyses That:**:
- Mention movies or attributes in Analysis but omit them in User Preferences.
- Ignore significant clues from the interaction history.
- Introduce unsupported preferences.
- Use the wrong format (missing structure, extra punctuation, non-English).
- Are overly short or generic in their summaries.

---

## Input Format:
### 1. User Interaction Record
A full interaction sequence, shown in turn-based format.

### 2. Multiple User Preference Analyses
Each preference analysis includes:
Preference Analysis [Id]:

### Analysis:
[A step-by-step reasoning based on the user interaction]

### User Preferences:
- movies:
  - like: list of liked movies or None
  - dislike: list of disliked movies or None
- attributes:
  - like: list of preferred genres, styles, directors, themes, stars, etc., or None
  - dislike: list of disliked genres, styles, directors, themes, stars, etc., or None

---

## Output Format:
### Reason:
Provide your reasoning for ranking. Focus on coverage, grounding, hallucination, relevance and the format.

Ranked Preference Analyses:
Rank the analyses from best to worst. Assign each a score from 1 to 5, where:
**5 = Excellent**: Strong alignment, detailed, grounded, well-formatted, comprehensive
**1 = Poor**: hallucinated, missing key preferences, off-topic, non-English or unstructured
Use this format:

### Ranked Preference Analyses:
1. Preference Analysis [Id], Score: [score]
2. Preference Analysis [Id], Score: [score]
...

---

**Important rules**:
- Do not use markdown symbols (*) to identify preference analysis ids and scores when listing them.
""".strip()

RECOMMEND_INSTRUCTION_SCORE_NO_TEXT = """
## Instruction
You are an expert in evaluating user preference analyses for movie recommendation systems.
You will be given the following:
1. A **complete user interaction history**, consisting of **only movie titles grouped by interaction**.
2. Several **user preference analyses**, each consisting of:
   - A step-by-step reasoning process (**Analysis**) based on the movie interactions
   - A structured summary (**User Preferences**) listing inferred movie preferences and attributes

---

### Your Task:
Your goal is to **score and rank each user preference analysis** based on how well it satisfies the following conditions:

1. **Coverage and Faithfulness**:
   - The `movies -> like` list **must include all movies** explicitly mentioned in the interaction history.
   - The `movies -> like` list must not include hallucinated or unmentioned movies.
   - The `movies -> dislike` list must be `None`, as disliking cannot be inferred from movie-only interactions.
   - The `attributes -> like` list **should include relevant attributes (genres, directors, actors, styles, etc.)** that can be **reasonably inferred from the listed movies.**
   - The `attributes -> dislike` list **must be** `None` due to lack of dialogue evidence.

2. **Reasoning Quality**:
   - The entire analysis and preference summary **must be in English only**.

3. **Output Format**:
Example of a valid output:
### User Preferences:
- movies:
  - like: The Dark Knight (2008), Titanic (1997)
  - dislike: None
- attributes:
  - like: action, drama, Christopher Nolan, Leonardo DiCaprio
  - dislike: None

4. **Penalize Analyses That:**:
- **Omit any mentioned movie from the `like` list**.
- **Omit any attributes that can be inferred from the `like` list**.
- Include any content in dislike fields

---

## Input Format:
### 1. User Interaction Record
A numbered list of movie-only interactions.

### 2. Multiple User Preference Analyses
Each preference analysis includes:
Preference Analysis [Id]:

### Analysis:
[A step-by-step reasoning based on interaction movies]

### User Preferences:
- movies:
  - like: list of liked movies or None
  - dislike: list of disliked movies or None
- attributes:
  - like: list of preferred genres, styles, directors, themes, stars, etc., or None
  - dislike: list of disliked genres, styles, directors, themes, stars, etc., or None

---

## Output Format:
### Reason:
Explain your ranking decisions. Focus on:
- Whether all mentioned movies are included (and none hallucinated)
- Whether attribute inferences are comprehensive, reasonable and traceable

Ranked Preference Analyses:
Rank the analyses from best to worst. Assign each a score from 1 to 5, where:
**5 = Excellent**: All mentioned movies included, valid attribute inferences
**1 = Poor**: missing movies, missing attributes, add movies / attributes to `dislike`
Use this format:

### Ranked Preference Analyses:
1. Preference Analysis [Id], Score: [score]
2. Preference Analysis [Id], Score: [score]
...

---

**Important rules**:
- Do not use markdown symbols (*) to identify preference analysis ids and scores when listing them.
""".strip()

RECOMMEND_INSTRUCTION_QWEN_MATCH_FINAL = """
## Instruction

You are an expert in analyzing user preferences based on their movie interactions. The following records include conversations and clicks, and they **belong to the same user**. Your task is to **think step by step** to infer this user's movie preferences.  
If the records do not provide enough information to infer preferences, respond with: "I don't know."

### **Important:**  
There are two parts to your output:
1. **Analysis**:  
   - Provide a step-by-step breakdown of how the user's interactions reveal their preferences.  
   - Follow the order of interaction records and keep the analysis **brief and clear**.
2. **User Preferences**:  
   - Summarize the user's preferences in a **structured format** with two main categories: `movies` and `attributes`.  
   - `movies` refers to specific movies or series the user likes or dislikes.  
   - `attributes` refers to general movie characteristics such as genres, themes, styles, directors, writers, stars, plots, etc.
   - If there is **insufficient information** to infer preferences for a category, output **None** for that category (e.g., `movies: like: None` or `attributes: like: None`).

### Input Format:
User interaction records are structured as follows:
[interaction Id]. 
[content]

## Output Format:
### Analysis:
[Follow the interaction records **in order** and provide a **brief and clear** step-by-step analysis. The analysis shouldn't be too long.]

After completing the step-by-step analysis, provide a final preference summary using **only the following format**:
### User Preferences:
- movies:
  - like: list of movies or series the user enjoys, or None if not enough information
  - dislike: list of movies or series the user dislikes, or None if not enough information
- attributes:
  - like: list of preferred attributes such as genres, themes, styles, directors, stars, etc., or None if not enough information
  - dislike: list of disliked attributes, or None if not enough information

For example:
- If the user likes the two movies Bridesmaids (2011) and Black Panther (2018). You should reply in this form:
### User Preferences:
- movies:
  - like: Bridesmaids (2011), Black Panther (2018)
  - dislike: None
- attributes:
  - like: comedy, romance, action, adventure, sci-fi
  - dislike: None
""".strip()

RECOMMEND_INSTRUCTION_QWEN3_MATCH_FINAL = """
## Instruction

You are an expert in analyzing user preferences based on their movie interactions. The following records include conversations and clicks, and they **belong to the same user**. Your task is to **think step by step** to infer this user's movie preferences.  
If the records do not provide enough information to infer preferences, respond with: "I don't know."

### **Important:**  
Your output should contain:
**User Preferences**:  
  - Summarize the user's preferences in a **structured format** with two main categories: `movies` and `attributes`.  
  - `movies` refers to specific movies or series the user likes or dislikes.  
  - `attributes` refers to general movie characteristics such as genres, themes, styles, directors, writers, stars, plots, etc.
  - If there is **insufficient information** to infer preferences for a category, output **None** for that category (e.g., `movies: like: None` or `attributes: like: None`).

### Input Format:
User interaction records are structured as follows:
[interaction Id]. 
[content]

## Output Format:
Provide a final preference summary using **only the following format**:
### User Preferences:
- movies:
  - like: list of movies or series the user enjoys, or None if not enough information
  - dislike: list of movies or series the user dislikes, or None if not enough information
- attributes:
  - like: list of preferred attributes such as genres, themes, styles, directors, stars, etc., or None if not enough information
  - dislike: list of disliked attributes, or None if not enough information

For example:
- If the user likes the two movies Bridesmaids (2011) and Black Panther (2018). You should reply in this form:
### User Preferences:
- movies:
  - like: Bridesmaids (2011), Black Panther (2018)
  - dislike: None
- attributes:
  - like: comedy, romance, action, adventure, sci-fi
  - dislike: None
""".strip()

RECOMMEND_INSTRUCTION_ATTITUDE = """
## Instruction

You are given a JSON object about movie discussion. The "text" field includes movie-related entities marked with <<>>, and these entities are listed in the "mapping" field.

### Task:
1. Analyze the user's attitude toward each entity in the "text" and classify it as **positive**, **neutral**, or **negative**.
2. Add a new key-value pair to the JSON: "sentiment": {entity: attitude, ...}. **Do not embed the sentiment within the "mapping" field.**
3. Return the updated JSON with the sentiment annotations. You don't have to include any instructions in your answer other than the complete updated JSON.

### Example:
If the json given to you is:
{"text": "Oh, <<The Wedding Singer (1998)>> is a good one.  Sandler and Barrymore are great together.  I'm very excited to finally see <<Blended  (2014)>>", "senderWorkerId": 4, "conversationId": "405", "timeOffSet": 3, "mapping": {"The Wedding Singer (1998)": 157394, "Blended  (2014)": 91313}}
You should return:
{
    "text": "Oh, <<The Wedding Singer (1998)>> is a good one.  Sandler and Barrymore are great together.  I'm very excited to finally see <<Blended  (2014)>>", 
    "senderWorkerId": 4, 
    "conversationId": "405", 
    "timeOffSet": 3, 
    "mapping": {"The Wedding Singer (1998)": 157394, "Blended  (2014)": 91313}, 
    "sentiment": {"The Wedding Singer (1998)": "positive", "Blended  (2014)": "positive"}
} 

### Note:
- The "sentiment" field must be a standalone key-value pair in the JSON, not nested within the "mapping" field.
- Focus only on the entities marked with <<>> and ensure the sentiment is one of the three options: **positive**, **neutral**, or **negative**.
""".strip()

RECOMMEND_INSTRUCTION_ANNOTATION = """
## Instruction

You are given a dialog JSON containing a movie recommendation conversation. Each movie title is replaced with @<id>. Your task is to:
1. Identify **movie properties**(e.g., genre, actor, director, writer, company) mentioned in the "messages" text.
2. Mark these properties with <<>> (e.g., <<comedy>>, <<Jim_Carrey>>).
3. You don't have to include any instructions in your answers other than the complete annotated json.
4. Return the **fully annotated JSON**, including original fields including "conversationId" and "initiatorWorkerId".

- Except for adding <<>> tags to words that appear in the original text, don't change any 'messages' text!

### Rules:
- Only mark properties that **explicitly appear in the text**. Do not add any properties that are not mentioned.
- **Do not modify or replace any @<id> placeholders.**
- If the json given to you doesn't have the required movie properties, simply return the original json.

### Important Note:
- Only mark properties that appear in the text field. Do not insert any additional words or attributes that are not present in the original text.
- Do not add any new content or explanations for @<id> placeholders. Only mark what is explicitly mentioned in the text.
""".strip()

CONVERSATION_TEMPLATE = """
## User interaction records:

{dialog_str}
""".strip()

CONVERSATION_TEMPLATE_SCORE = """
## 1. User Interaction Record

{dialog_str}
""".strip()