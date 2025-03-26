# 对于Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8模型

## 1. 测试prompt：
```
## Instruction

You are an expert in analyzing user preferences based on their movie interactions. The provided interaction records **belong to the same user**. Your task is to **think step by step** to infer this user's movie preferences. 
If the records do not provide enough information to infer preferences, respond with: "I don't know."

### Input Format:
User interaction records are structured as follows:
[interactionId]. [action]
[content]

## Analysis Format:
Follow the interaction records **in order** and analyze the user's preferences step by step:
[interactionId]. [action]:
- (Summarize key details from this interaction.)
- (Extract any stated or implicit preferences.)
[interactionId]. [action]:
- (Summarize key details from this interaction.)
- (Extract any stated or implied preferences.)
...

## Output Format:
After completing the step-by-step analysis, provide a final preference summary using **only the following format**:
### User Preferences:
- (Summarized movie preferences, based on inferred attributes such as **genres, themes, actors, directors, writers, plots, or other relevant aspects.**)

## Attention:
1. If there is **insufficient information** to infer preferences, respond with:
### User Preferences: I don't know.

2. **Do not mention specific movie titles** in the final **User Preferences** summary.

## User interaction records:
...
```

问题分析：
1. Analysis Format部分引导模型逐个交互记录进行分析，部分交互记录多达100个，模型分析部分过长且冗余。
2. 未告知模型Analysis Format应该是Output Format的一部分，未明确最终输出应该按照 ### Analysis + ### User Preferences的形式呈现。

- 通过这一版prompt观察到Qwen对“不在User Preferences中携带电影名称”的提示遵从较好。

- 直接放弃该版prompt生成的COT文本。

----------------------------------------------------------------------------------------------------------------------------------------

## 2. 目前效果较好的prompt：
```
## Instruction

You are an expert in analyzing user preferences based on their movie interactions. The provided interaction records **belong to the same user**. Your task is to **think step by step** to infer this user's movie preferences. 
If the records do not provide enough information to infer preferences, respond with: "I don't know."
**Do not include specific movie titles or series names in your final response.**

### Input Format:
User interaction records are structured as follows:
[interactionId]. [action]
[content]

## Analysis Format:
Follow the interaction records **in order** and analyze the user's preferences step by step.
- (Summarize key details from interaction records.)  
- (Extract any stated or implicit preferences.) 

## Output Format:
### Analysis:
[step-by-step analysis]

After completing the step-by-step analysis, provide a final preference summary using **only the following format**:
### User Preferences:
- (Summarized movie preferences, based on inferred attributes such as **genres, themes, actors, directors, writers, plots, or other relevant aspects.**)
- (Each preference should be on a separate line.)
- **(Ensure no movie titles appear in the response. If they do, remove them before submitting.)**

## Attention:
1. If there is **insufficient information** to infer preferences, respond with:
### User Preferences: I don't know.

2. **Do not mention specific movie titles** in the final **User Preferences** summary, including franchise or series names such as **Star Wars, Harry Potter, The Lord of the Rings**, etc.

## User interaction records:
...
```

大部分回复令人满意，模型也理解要求输出的结构，因此没有测试直接将Analysis Format作为Output Format来简化prompt

1. 
较短交互记录的分析：
```
### Analysis:
- The user explicitly states a preference for sci-fi and horror genres.
- The user also mentions enjoying movies that make them think, indicating a preference for intellectually stimulating content.
- The user mentions liking "District 9 (2009)" and "The Sixth Sense (1999)," which are examples of sci-fi and horror movies, respectively, that also have elements that make the viewer think.

### User Preferences:
- Prefers sci-fi and horror genres.
- Enjoys movies that are intellectually stimulating and thought-provoking.
```

2. 
较长交互记录的分析：
```
### Analysis:
- The user expresses a preference for comedy movies, mentioning classics like Home Alone (1990) and Will Ferrell's movies.
- The user also shows interest in action movies, specifically mentioning classics like The Transporter (2002) and newer ones like Black Panther (2018).
- The user has a strong interest in Black Panther (2018), indicating a preference for superhero movies.
- The user also enjoys thrillers and suspense movies, such as The Hangover (2009) and Saw (2004).
- The user has shown interest in a variety of genres, including action, comedy, and thrillers, and has interacted with recommendations for movies like The Mummy (2017) and The Italian Job (1969).
- The user has also expressed interest in sci-fi movies, such as The Mummy (2017) and Guardians of the Galaxy (2014).
- The user has shown a preference for movies with strong action elements, such as Die Hard (1988) and The Terminator (1984).
- The user has also shown interest in classic comedies like Pulp Fiction (1994) and Ferris Bueller's Day Off (1986).

### User Preferences:
- Action movies with strong plot and character development.
- Comedy movies, especially classics and those featuring Will Ferrell.
- Superhero movies, particularly those with a strong narrative and character focus.
- Thrillers and suspense movies with a mix of humor.
- Sci-fi movies with a blend of action and comedy.
- Classic movies with enduring appeal and strong performances.
```

- 仍然会有部分回复在User Preferences中携带电影名称，主要集中在(提到Star Wars / The Hobbit系列，或在说dislike...时)：

1. 
```
### User Preferences:
- Prefers modern comedies.
- Dislikes romantic elements in comedies.
- Dislikes certain actors or styles present in "Central Intelligence (2016)" and "Rush Hour (1998)."
- Open to discovering new comedies.
```

2. “Star Wars系列”很容易两种模型在回复中多次提及
```
- Prefers science fiction movies, particularly those from the Star Wars franchise.
- Enjoys movies with strong action and adventure elements.
- Not interested in superhero movies from the Marvel Cinematic Universe.
- Likes movies that are suitable for children and can be enjoyed with kids.
```
3. 
```
- Prefers family-friendly and animated films.
- Enjoys fantasy and young adult themes.
- Interested in films that teach positive social values.
- Has a preference for certain entries within the Star Wars franchise.
- Not interested in "The Avengers" series.
```

- 可以显著减少grad_norm训练过程中出现的尖刺。
- User Preferences中的tokens比较少，最多239个tokens（训练稳定和文本整体上更短有关？）

在分析tokens较多的User Preferences时，发现还是有小部分User Preferences携带了电影名称，因此考虑使用更严格的prompt。
考虑对要求输出的两部分更详细的说明。

## 3. 更严格的prompt：
```
## Instruction

You are an expert in analyzing user preferences based on their movie interactions. The provided interaction records **belong to the same user**. Your task is to **think step by step** to infer this user's movie preferences. 
If the records do not provide enough information to infer preferences, respond with: "I don't know."

### **Important:**  
There are two parts to your output:
1. **Analysis** - In this part, you can refer to specific movie titles or series names to help identify user preferences step by step.  
2. **User Preferences** - In this part, **do not mention specific movie titles or series names**. Instead, generalize any references to movies into their **characteristics** (e.g., genre, themes, narrative styles, etc.). Do not justify preferences by mentioning movie titles.

### Input Format:
User interaction records are structured as follows:
[interactionId]. [action]
[content]

## Analysis Format:
Follow the interaction records **in order** and analyze the user's preferences step by step.
- (Briefly summarize key points from the interactions.)  
- (Identify explicit or implicit preferences based on the interaction details.) 

## Output Format:
### Analysis:
[step-by-step analysis]

After completing the step-by-step analysis, provide a final preference summary using **only the following format**:
### User Preferences:
- (Summarized movie preferences, based on inferred attributes such as **genres, themes, actors, directors, writers, plots, or other relevant aspects.**)
- **Do not use any specific movie titles or series names at any point in the user preferences summary**.
- **Convert any references to movies into attribute preference**, such as **genres, themes, or narrative styles**.  
- Ensure each preference is on a separate line for clarity.

## Attention:
1. If there is **insufficient information** to infer preferences, respond with:
### User Preferences: I don't know.

2. **Do not mention specific movie titles** in the final **User Preferences** summary, including franchise or series names such as **Star Wars, Harry Potter, The Lord of the Rings**, etc.

3. **In the Analysis section**, movie titles can be mentioned, but in the **User Preferences** summary, they should be generalized to their features.

## User interaction records:
...
```

复杂的写法，尤其是：
**Convert any references to movies into attribute preference**
Ensure each preference is on a separate line for clarity.
这样的强制转换，容易导致模型输出风格固定、重复输出一些相同的内容。

现阶段的较长的输出存在这样的形式：
```
- Prefers films with musical elements, particularly those that revolve around music and musicians.
- Enjoys films that incorporate singing without being full-fledged musicals.
- Interested in films that explore themes of pregnancy and family.
- Prefers films with crime and action elements, such as heist movies.
- Enjoys well-received and critically acclaimed dramas.
- Prefers light-hearted and silly comedies.
- Interested in science fiction films that explore alien themes.
- Enjoys intense and dramatic narratives, particularly those with revenge themes.
- Prefers thrillers and horror films that explore themes of serial killers.
- Interested in films that explore mental health themes, such as depression.
- Prefers science fiction films that explore future settings.
- Enjoys films that are suitable for young audiences.
- Interested in films that explore friendship themes.
- Prefers films that feature young protagonists in adventurous settings.
- Enjoys films that incorporate music in a non-musical context.
- Prefers films based on Stephen King's works.
- Enjoys films that explore themes of magic and illusion.
- Interested in fantasy and adventure films, particularly those that are part of larger franchises.
- Enjoys horror films that explore scary themes.
- Prefers war and historical dramas.
- Enjoys films that are part of larger franchises.
- Prefers romantic comedies that explore romantic themes.
- Interested in films that are either made in Italy or set in Italy.
- Enjoys action and adventure films, particularly those that are part of larger franchises.
```

## 4. 改进想法
1. 不对输出格式进行限制，模型自由输出一段话即可
2. 可能需要对历史记录进行一下截断，避免“The user prefers a mix of genres including...”这样过于宽泛的偏好总结
3. 看看模型对Conversation和Click两种不同交互方式的反映（有些回复中模型更关注对话），可以为后期的文本替换位置实验提供帮助

## 4.1 不对格式输出进行引导：
```
## Instruction

You are an expert in analyzing user preferences based on their movie interactions. The provided interaction records **belong to the same user**. Your task is to **think step by step** to infer this user's movie preferences. 
If the records do not provide enough information to infer preferences, respond with: "I don't know."

### **Important:**  
There are two parts to your output:
1. **Analysis** - In this part, you can refer to specific movie titles or series names to help identify user preferences step by step.  
2. **User Preferences** - In this part, **do not mention specific movie titles or series names** (This includes avoiding exceptions like "except for "[movie name]."). Instead, you should summarize user preferences for movie attributes (e.g., genre, themes, narrative styles, etc.). Do not justify movie preferences by mentioning movie titles.

### Input Format:
User interaction records are structured as follows:
[interactionId]. [action]
[content]

## Output Format:
### Analysis:
[Follow the interaction records **in order** and analyze the user's preferences step by step.]

After completing the step-by-step analysis, provide a final preference summary using **only the following format**. Ensure no movie titles or series names like "Star Wars", "Harry Potter" etc. appear in the response. If they do, remove them before submitting.
### User Preferences:
[Summarized movie preferences, based on inferred attributes such as **genres, themes, actors, directors, writers, plots, or other relevant aspects.**]

## Attention:
1. If there is **insufficient information** to infer preferences, respond with:
### User Preferences: I don't know.

2. **Do not mention specific movie titles** in the final **User Preferences** summary, including franchise or series names such as **Star Wars, Harry Potter, The Lord of the Rings**, etc.

3. **In the Analysis section**, movie titles can be mentioned, but in the **User Preferences** summary, they should be generalized to their features.

## User interaction records:
...
```

模型的输出形式：
1. 
```
### Analysis:
The user has shown a strong preference for romantic comedies, particularly those featuring actors like Drew Barrymore, Reese Witherspoon, and Will Ferrell. They have also expressed interest in superhero movies, especially those with strong female leads, and have shown enjoyment in animated films, particularly those with musical elements. The user has also shown interest in action movies, especially those in the spy genre, and has a fondness for Christmas-themed movies and classics. They have a particular interest in comedies, including those with a more adult-oriented humor, and have shown appreciation for musicals and family-friendly films.

### User Preferences:
- Prefers romantic comedies and comedies in general.
- Enjoys films with strong female leads.
- Likes animated films, especially those with musical elements.
- Appreciates action movies, particularly in the spy genre.
- Enjoys Christmas-themed movies and classics.
- Has a fondness for family-friendly films and musicals.
- Enjoys comedies with adult-oriented humor.
```

2. 
```
### Analysis:
The user expresses a clear preference for sci-fi and comedy genres. They mention being a huge fan of the Star Wars franchise, which indicates a strong interest in sci-fi movies with epic storytelling and action. Additionally, the user mentions liking Kevin Hart, suggesting a preference for comedic actors and films that feature his style of humor. The user also expresses enjoyment of "Ride Along 2," which is a comedy film, further supporting their preference for comedy.

### User Preferences:
The user prefers movies in the sci-fi and comedy genres, with a particular interest in epic storytelling and action within sci-fi films. They also enjoy comedic actors and films that feature light-hearted humor.
```

3. 感到比较疑惑的回复：
```
## User interaction records:

1. Conversation
System: Hello
System: what types of movies do you enjoy
User: Hi I like sci fi and comedy movies am a huge Star Wars fan and love Kevin Hart
System: great, have you seen Ride Along 2 (2016) yet?
User: Yeah I loved that movie.
User: I also loved Central Intelligence (2016)
System: Cool, you might like @bluestreak with Martin Lawrence
User: I have not seen that one what is it about
User: Blue Streak  (1999) does seem like it would be funny
System: hmmm...i added that but its not showing up. its hilarious. he was a crook and after jail becomes a police officer to get back a hidden diamond he left before jail
System: oohh thanks
User: Wow that sounds very interesting I have to watch that.
System: also, Bulletproof  (1996) is anothe you may like to check out
System: Adam Sandler and Damon Wayans
User: Wow thanks those both sound like great movies
System: no problem, have a good day! bye
User: Thanks you too. Bye

2. Conversation
System: hey there
User: Hi what kind of movies do you like
System: id like to recommend some movies for you. what are you interested in?
System: I like older stuff like Risky Business  (1939)
LLM Response:
### Analysis:
The user expresses a clear preference for sci-fi and comedy genres, indicating a fondness for movies like those in the Star Wars franchise and films starring Kevin Hart. They also enjoyed "Ride Along 2" and "Central Intelligence," both of which are comedy films. The user shows interest in action-comedy films, as evidenced by their positive reaction to the plot summary of "Blue Streak" and their eagerness to watch "Bulletproof," another action-comedy film. The user's interest in actors like Kevin Hart, Martin Lawrence, Adam Sandler, and Damon Wayans suggests a preference for comedic actors. However, the second conversation snippet seems to contain an error, as "Risky Business" was released in 1983, not 1939, and the user's statement about liking older movies is not consistent with the other interactions which focus on more recent films.

### User Preferences:
I don't know.
```
- 模型在User Preferences总结中输出 I don't know.的概率增加

- grad norm在下降初期存在大量尖刺，且发生梯度爆炸 -> 与数据格式有关？

# 对于Deepseek模型:

```
## Instruction

You are an expert in analyzing user preferences based on their movie interactions. The provided interaction records **belong to the same user**. Your task is to infer this user's movie preferences. 
If the records do not provide enough information to infer preferences, respond with: "I don't know."

**Strict Rules (Do Not Violate These Rules)**
1. **Do not include specific movie titles or series names** (e.g., Star Wars, E.T., Harry Potter, Marvel movies, etc.) in your final response.  
2. **Do not justify preferences by mentioning movie titles.** Instead, describe characteristics such as **genre, themes, narrative style, or notable actors/directors.**  
3. **Before submitting your response, verify that it does not contain any movie titles.** If any exist, **remove them**. 

### Input Format:
User interaction records are structured as follows:
[interactionId]. [action]
[content]

## Output Format:
After completing the analysis, provide a final preference summary using **only the following format**:
### User Preferences:
- (Summarized movie preferences, based on inferred attributes such as **genres, themes, actors, directors, writers, plots, or other relevant aspects.**)
- (Each preference should be on a separate line.)
- **(Ensure no movie titles appear in the response. If they do, remove them before submitting.)**  

## Attention:
1. **Never include movie titles** from the user's interaction records in your response.

2. If there is **insufficient information** to infer preferences, respond with:
### User Preferences: I don't know.

3. **List each preference category separately** (e.g., genre, themes, actor preferences, etc.), ensuring clarity.

4. **Before finalizing your response, double-check and remove any remaining movie titles.**

## User interaction records:
...
```

模型往往会在reasoning content中携带：
I need to make sure I don't mention any specific movie titles, so I'll focus on the genres, themes, and types of movies they've shown interest in.

但是content输出中仍然携带：
- Enjoys action thrillers, particularly those featuring maverick cops like the Dirty Harry series.
- Prefers superhero movies, especially those with strong female leads and epic narratives.
- Appreciates romantic dramas and classic romantic comedies, particularly from the 60s and 70s.
- Enjoys animated films, especially Disney and Pixar classics with strong themes.
- Interested in sci-fi action, including superhero and space adventure themes.
- Enjoys comedy, both slapstick and character-driven.
- Likes military and espionage thrillers, such as The Hunt for Red October.
- Enjoys classic films, both in comedy and drama genres.
- Appreciates animated sci-fi and adventure films like WALL-E and Coco.

模型很喜欢利用电影名称佐证用用户的某项偏好，且不包含电影名称的输出和Qwen生成的User Preferences并无太大差别，因此简易测试后不再使用。