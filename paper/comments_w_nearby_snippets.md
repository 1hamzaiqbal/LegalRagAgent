Legal RAG can fail at query formation, evidence exposure, or answer conversion;
a single accuracy number hides which one happened. \hll{Need some background introduction, like HyDE} We evaluate \hyre{}, a fixed

also add more reference to the introduction section

----
\item We assemble a legal RAG evaluation grid with explicit coverage annotations
over four legal benchmarks and three hosted model configurations.

comment: shoudl be do some experiments


\item We separate retrieval exposure from answer conversion. The audited rows
show BarExamQA retrieval/answer gains, HousingQA jurisdiction-interface
sensitivity, Legal-Link-EU retrieval degradation under generated queries with a
Gemma 4 26B \hyre{} parity slice, MASLegalBench source-proxy saturation, and
worked examples where generated legal passages repair raw-query misses.

give specific performance numbers here -- find max improvement, then 'up to'

\item We package the evaluation with oracle controls, coverage status, and
paper-facing qualification notes.
\end{enumerate}

comment: mention some meaningful analysis from paper (analytical study). overall make contribution points seem useful and concrete


----
\paragraph{Generated-query retrieval and HyDE.}
HyDE generates a hypothetical document from a query and embeds that document

comment: Keep background clean of snap hyre mention, talk about snap hyre in intro, then hyde + older methods independent of snap hyre in this section

\paragraph{Evaluation stance.}

comment: Put this in experiment section

----
\begin{table*}[t]
\centering
\caption{Dataset interfaces used by the canonical methods. The row definition
is fixed across datasets; answer format, corpus, labels, and required metadata
filters are dataset properties.}

comment: Overlaps with section 4.1 -- can remove, also can merge and shorten benchmark exposition

---

once method gets shortened or things go in there, then you add more design and intuition about why we doing snap hyre


\subsection{Snap-HyRE Mechanics}\label{sec:method:snaphyre}

comment: maybe here, add motivation of each step, write some words to show intentional design of every
Hamza Iqbal
20 May, 4:31 pm
maybe show example to show *why* we do things, snap or without snap, make narrative better and less ...
Hamza Iqbal
20 May, 4:32 pm
look at hyde paper, think about reasonable equations to add to this portion https://arxiv.org/pdf/2212.10496


\subsection{Metrics}

comment: shoudl be in experiment setup

----
\section{Experimental Setup}\label{sec:experiment}

comment: Main method: main idea, highlight the imp stuff (rag simpl, rag hyde, big important need to know), NOT golden passage, rag rewrite, or other less important methods). Experiments: specific configurations , also table 2 dataset descriptions shoudl be in expt setup

The exact-scored main matrix contains four legal benchmarks:
BarExamQA, HousingQA, Legal-Link-EU, and MASLegalBench.

The experiment is designed around three questions. First, when does a
snap-conditioned generated legal passage improve retrieval exposure over the
raw question? Second, when does that exposure convert into final answer
accuracy? Third, when do oracle evidence controls show that retrieval is no
longer the binding failure point?

comment: benchmark exposition long


----
A few conventions matter before reading the tables. Asterisks mark unfiltered

comment: overall things need to do, also rename secitons to be straighhtforward
Main Results
Reply
Hamza Iqbal
20 May, 4:50 pm
Reorg full paper -- 
Hamza Iqbal
20 May, 4:50 pm
figure 1 redraw, methods rewrite, section header better, stronger tables


----
  \caption{Retrieval accuracy versus answer accuracy delta, both relative
  to raw RAG. Points above zero on the $y$ axis improve final accuracy; points
  right of zero improve Hit@5 or same-source@5. MASLegalBench points use the

  comment: Use dotted line to show baseline, put zero point at bottom left part. 
Reply
Hamza Iqbal
20 May, 4:52 pm
Baseline is dotted line (methods worked better on retrieval vs better on final accuracy)
Reply
Hamza Iqbal
20 May, 4:54 pm
When does retrieval accuracy help downstream accuracy? (this is a better title)


----
\section{Appendix}\label{sec:appendix}

every part of appendix needs to expand on some aspect discussed in main portion of paper

Oracle evidence controls relative to raw RAG.

comment: usually 2-3 additional analyses goes in appendix. see if most important/relevant analyses and put in main text (something like ablation study or top 1-10 (top k variation), OR good quantitative analyses
Hamza Iqbal
20 May, 5:03 pm
appendix can be referenced in main text to allude to less narrative important but informative results