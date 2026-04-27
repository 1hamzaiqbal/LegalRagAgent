# Methods Explained

## `rag_simple`

`rag_simple` is the control retrieval-augmented generation path: retrieve passages with the original MuSiQue question, place those passages in the answer prompt, and ask the Llama 70b model for the final answer. We tested it because every method in the matrix needs a stable retrieval baseline; otherwise lifts can be confounded by model size, provider, or dataset slice. On Llama 70b N=200, this baseline gets 27.5% EM and is marked CLEAN and APPROVED in Table D. The example is intentionally plain: no snap answer, no HyDE passage, no planning table, and no subagent report. This is the row that the other seven methods must beat under paired McNemar.

Source: logs/eval_rag_simple_groq-llama70b_20260427_0952_detail.jsonl:6.

```json
{
  "question": "Along with the Closer performer, what notable pop artist started out his career on adult contemporary radio?",
  "final_answer": "Answer: Michael Bublé",
  "predicted_answer": "Michael Bublé",
  "correct_answer": "Michael Bublé",
  "is_correct": true
}
```

## `multi_hyde_diverse`

`multi_hyde_diverse` generates multiple hypothetical answer-bearing passages, then uses those passages as extra retrieval signals before the final answer step. The intuition is that multi-hop questions often fail because the original question underspecifies the bridge entity; diverse answer-shaped passages can name plausible entities and pull more useful evidence. This is the Llama 70b paper headline: 35.5% EM, +8.0pp over `rag_simple`, McNemar p=0.0195, APPROVED. The example shows why the method helps when the generated passages land on the right bridge: both HyDE passages identify Matt Damon/Jason Bourne before the model answers.

Source: logs/eval_multi_hyde_diverse_groq-llama70b_20260427_1010_detail.jsonl:11.

```json
{
  "question": "Who is the actor who plays the title character of The Bourne Ultimatum?",
  "hyde_passages": [
    "The actor who plays the title character of The Bourne Ultimatum is Matt Damon, an American actor known for his portrayal of Jason Bourne in the Bourne film series. Damon's performance in the series has been widely praised, and he has become closely associated with the role of the amnesiac assassin. His depiction of Bourne's struggles with identity and morality has been a key factor in the series' success.",
    "The role of Jason Bourne has also been influenced by the direction of Paul Greengrass, who helmed The Bourne Ultimatum and The Bourne Supremacy, and worked closely with Matt Damon to develop the character. Greengrass's kinetic filmmaking style and emphasis on realistic action sequences have helped to shape the tone and pace of the series, and his collaboration with Damon has been instrumental in creating a compelling on-screen presence for the character. The success of the Bourne series can be attributed in part to the strong working relationship between Greengrass and Damon."
  ],
  "final_answer": "Answer: Matt Damon",
  "predicted_answer": "Matt Damon",
  "correct_answer": "Matt Damon",
  "is_correct": true
}
```

## `iterative_planning_table`

`iterative_planning_table` turns the question into a sequence of TODOs, retrieves evidence for each TODO, records findings, and then asks the final model to synthesize the chain. We tested it because multi-hop failures often look like missing intermediate state: the model either skips a bridge or answers before checking a needed entity. This method reaches the best raw EM in the Llama 70b matrix, 36.0%, with +8.5pp over baseline. It is APPROVED as TRENDING-SIG rather than fully significant because p=0.0533. The example is a clean pass: the table first identifies Mac Miller as the relevant rapper, then separately finds the release date for `Best Day Ever`.

Source: logs/eval_iterative_planning_table_groq-llama70b_20260427_1208_detail.jsonl:13.

```json
{
  "question": "When did the rapper on On and On and Beyond release Best day Ever?",
  "planning_table": [
    {
      "todo": "Who is the rapper that released the songs \"On and On\" and \"Beyond\"?",
      "finding": "The passages do not contain information about a rapper releasing the songs \"On and On\" and \"Beyond\". The title \"On and On and Beyond\" is mentioned as the debut extended play by American rapper Mac Miller, but it does not specify that the songs \"On and On\" and \"Beyond\" are part of the EP."
    },
    {
      "todo": "When did Mac Miller release Best Day Ever?",
      "finding": "Mac Miller released Best Day Ever on March 11, 2011. This mixtape was released online and was followed by a live video stream. Over 20,000 viewers joined the stream just prior to the release."
    }
  ],
  "final_answer": "To answer the question, we need to identify the rapper associated with \"On and On and Beyond\" and then determine when that rapper released \"Best Day Ever\".\n\n1. From the Iterative Planning Table, Round 1, we find that the debut extended play by American rapper Mac Miller is titled \"On and On and Beyond\". This implies that Mac Miller is the rapper associated with \"On and On and Beyond\".\n\n2. Next, we look for information on when Mac Miller released \"Best Day Ever\". The Iterative Planning Table, Round 2, provides this information: Mac Miller released \"Best Day Ever\" on March 11, 2011.\n\nTherefore, by connecting the rapper Mac Miller to both \"On and On and Beyond\" and \"Best Day Ever\", we conclude that Mac Miller, the rapper associated with \"On and On and Beyond\", released \"Best Day Ever\" on March 11, 2011.\n\nAnswer: March 11, 2011",
  "predicted_answer": "March 11, 2011",
  "correct_answer": "March 11, 2011",
  "is_correct": true
}
```

## `rag_multi_query`

`rag_multi_query` asks the model to rewrite the original question into multiple retrieval queries, then pools evidence before final answering. We tested it to separate query diversity from HyDE-style answer-passage generation. If diverse retrieval alone explained the MuSiQue lift, this method should have moved close to `multi_hyde_diverse`; it did not. On Llama 70b N=200, it reaches 29.0% EM, only +1.5pp over baseline, with McNemar p=0.728. The example is a failure case: the rewrites ask broad NBA-draft questions, but they do not pin down the scoring-title winner, and the final answer lands on the wrong team.

Source: logs/eval_rag_multi_query_groq-llama70b_20260427_1112_detail.jsonl:2.

```json
{
  "question": "What team drafted the winner of the NBA scoring title this year?",
  "rewrites": [
    "Which NBA teams have been most successful in the draft regarding players who go on to achieve high scoring averages?",
    "What are the common characteristics of players who have won the NBA scoring title in recent years?"
  ],
  "final_answer": "Answer: Houston Rockets",
  "predicted_answer": "Houston Rockets",
  "correct_answer": "Oklahoma City Thunder",
  "is_correct": false
}
```

## `rag_snap_hyde`

`rag_snap_hyde` first asks for a quick answer, converts that snap answer into a HyDE-style passage, retrieves from that synthetic passage plus the question, and then performs final RAG. We tested it because it is the BarExam winner and we needed to know whether that legal-MC recipe transfers to multi-hop open-domain QA. It does not: on Llama 70b MuSiQue N=200, it drops to 24.0% EM, -3.5pp vs baseline, with p=0.36. The example shows the failure mode: the snap answer and HyDE passage confidently move into an irrelevant law, and the final answer abstains instead of recovering the target.

Source: logs/eval_rag_snap_hyde_groq-llama70b_20260427_1019_detail.jsonl:62.

```json
{
  "question": "What was the name of the law passed by the actor from Terminator?",
  "snap_answer": "Answer: AB 1881",
  "hyde_passage": "The California Legislature passed Assembly Bill 1881, which was signed into law by Governor Jerry Brown in 2016, and this law restricts state-funded travel to states that have enacted laws discriminating against LGBTQ individuals. This law, also known as AB 1881, is a California state law that applies to state agencies, departments, boards, and commissions, and it prohibits them from requiring their employees to travel to any state that has enacted a law that discriminates against individuals based on their sexual orientation, gender identity, or gender expression. The law is intended to promote equality and protect the rights of LGBTQ individuals by restricting state-funded travel to states with discriminatory laws.",
  "final_answer": "Answer: None",
  "predicted_answer": "None",
  "correct_answer": "\"Donda West Law\"",
  "is_correct": false
}
```

## `iter_hyde`

`iter_hyde` performs HyDE in rounds: generate a focused hypothetical passage, retrieve evidence, record a finding, and optionally continue with the next sub-investigation before final answering. The intuition was that repeated HyDE could repair missing bridges that one synthetic passage misses. On Llama 70b N=200, it does not: 24.5% EM, -3.0pp vs baseline, p=0.47. The example shows the problem in miniature. The generated chain correctly retrieves Oregon State University and the Beavers, but the final answer gives the team nickname rather than the mascot name required by the gold answer.

Source: logs/eval_iter_hyde_groq-llama70b_20260427_1036_detail.jsonl:142.

```json
{
  "question": "What is the mascot of the school that owns Goss Stadium at Coleman Field?",
  "hyde_chain": [
    {
      "hyde": "Goss Stadium at Coleman Field is a baseball stadium located in Corvallis, Oregon, and it is owned by Oregon State University. Oregon State University is a public research university that was founded in 1868. The university's athletic teams, known as the Oregon State Beavers, compete in the Pac-12 Conference.",
      "finding": "The sub-investigation focus is about Goss Stadium at Coleman Field, which is a college baseball stadium located in Corvallis, Oregon, and is home to the Oregon State Beavers. The stadium is owned by Oregon State University, a public research university founded in 1868. The university's athletic teams compete in the Pac-12 Conference."
    }
  ],
  "final_answer": "To determine the mascot of the school that owns Goss Stadium at Coleman Field, let's follow the information provided in the HyDE-finding chain:\n\n1. The stadium in question is Goss Stadium at Coleman Field.\n2. This stadium is owned by Oregon State University.\n3. Oregon State University's athletic teams are known as the Oregon State Beavers.\n\nGiven this chain of information, we can conclude that the mascot of the school that owns Goss Stadium at Coleman Field is the Beavers.\n\nAnswer: Beavers",
  "predicted_answer": "Beavers",
  "correct_answer": "Benny Beaver",
  "is_correct": false
}
```

## `advisor_planning_table`

`advisor_planning_table` also builds a planning table, but uses an advisor-style decomposition where the model proposes TODOs and findings before the final response. We tested it because planning is plausibly useful, but the exact planning prompt matters. The Llama 70b result is negative: 23.0% EM, -4.5pp vs baseline, p=0.222, approved as non-significant but informative. The example shows the failure mode: the table finds two local facts about `Study in Brown`, then fails to bridge to the explorer/date relation and returns no information rather than the gold date.

Source: logs/eval_advisor_planning_table_groq-llama70b_20260427_1216_detail.jsonl:123.

```json
{
  "question": "When did the explorer reach the headquarters location of the group Study in Brown's record label is part of?",
  "planning_table": [
    {
      "todo": "What is the name of the record label that Study in Brown is part of?",
      "finding": "Study in Brown is part of EmArcy Records."
    },
    {
      "todo": "What is the name of the group that Study in Brown is part of?",
      "finding": "The album \"Study in Brown\" is a Clifford Brown and Max Roach album."
    },
    {
      "todo": "What is the name of the explorer associated with Study in Brown?",
      "finding": "There is no information about an explorer associated with Study in Brown in the retrieved passages."
    }
  ],
  "final_answer": "To answer the question, we need to follow the multi-hop chain explicitly. \n\n1. The record label that \"Study in Brown\" is part of is EmArcy Records.\n2. \"Study in Brown\" is an album by the group Clifford Brown and Max Roach.\n3. However, there is no information about an explorer associated with \"Study in Brown\".\n\nGiven the lack of information about an explorer, we cannot determine when the explorer reached the headquarters location of the group \"Study in Brown's\" record label. Since we do not have any details about an explorer, we cannot establish a connection to a specific date or event related to reaching a headquarters location.\n\nAnswer: No information available",
  "predicted_answer": "No information available",
  "correct_answer": "August 3, 1769",
  "is_correct": false
}
```

## `subagent_rag`

`subagent_rag` tries to identify missing information gaps, run targeted subqueries, collect reports, and then use those reports in the final answer. We tested it because explicit gap-filling sounds like the right abstraction for multi-hop QA. Empirically, this implementation is the clearest negative result: 15.5% EM, -12.0pp vs baseline, McNemar p=0.0007. The example shows why it can hurt. The subagent asks a narrow league-format question and retrieves a report that even mentions 38 matchdays, but the final answer still commits to 26, the wrong number.

Source: logs/eval_subagent_rag_groq-llama70b_20260427_1044_detail.jsonl:89.

```json
{
  "question": "How many games in a season of the league in which Barcelona won titles in 1948 and 1949?",
  "gaps": [
    {
      "description": "the specific league structure or rules that might have applied to the 1947-1948 and 1948-1949 seasons of the league in which Barcelona competed",
      "sub_question": "Were there any exceptions or variations in the league format, such as a separate championship phase or playoff, that could have altered the total number of games played by Barcelona during those seasons?"
    }
  ],
  "gap_results": [
    {
      "retrieval_query": "Were there any exceptions or variations in the league format, such as a separate championship phase or playoff, that could have altered the total number of games played by Barcelona during those seasons?",
      "report": "The passages do not provide information about the league format or number of games played by Barcelona during the 1948 and 1949 seasons. However, Source 2 mentions that La Liga follows a double round-robin format with 38 matchdays, but it does not specify if this format was in place during the 1948 and 1949 seasons."
    }
  ],
  "final_answer": "Answer: 26",
  "predicted_answer": "26",
  "correct_answer": "38",
  "is_correct": false
}
```

Method-result source for all eight sections: docs/signoff_log.md Section B.1 and Table D in docs/presentation/01_results_tables.md.
