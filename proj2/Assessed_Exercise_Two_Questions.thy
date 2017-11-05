
section\<open>Introduction\<close>

text\<open>This is the second assessed exercise for the L21 ``Interactive Formal Verification'' MPhil
course in academic year 2017--2018.  In this exercise you will prove some properties of structures
called metric spaces and two closely associated concepts: continuity and open balls.  The exercise
will test your ability to write structured proofs, work with sets and quantifiers, and carefully
select relevant auxiliary lemmas and introduction and elimination principles to make theorem proving
easier.  Indeed, all proofs should be written as Isar structured proofs, as far as possible.
Concretely, the marking scheme for this exercise is as follows (out of a total of 100):

  \<^item> \textbf{50 marks} for correct definitions, lemma statements and structured proofs, in accordance
    with the distribution of marks outlined in the body of this document,
  \<^item> \textbf{30 marks} for extra proofs of properties about metric spaces, continuity and open balls
    presented in this exercise, more definitions related to metric spaces and properties about them,
    or the use of features of Isabelle not lectured in class, or similar along these lines.  Any
    reasonable evidence that you have gone ``above and beyond'' in your use of Isabelle will be
    considered here, and obviously the more ambitious you are, the more marks you accrue,
  \<^item> \textbf{20 marks} for a nice writeup detailing all decisions made in your choice of lemmas,
    proof strategies, and so on, and an explanation as to what extensions you have decided to
    implement, or novel features of Isabelle that you have used, to merit some of the 30 marks
    mentioned above.

Your submissions should be submitted electronically on the L21 Moodle site before 4pm Cambridge
local time on submission day.  See the course website for submission dates for the two assessed
exercises. Submissions should consist of an edited version of the theory file that this document is
based on (also available from the course website) that includes your solutions to the exercises
contained within, along with a PDF document containing your writeup.  Your writeup need not be
especially long---2 sides of A4 will suffice---but it should be detailed, describing auxilliary
lemmas and definitions that you introduced, if any, design decisions, and so on.  Late submissions
will of course be penalised, and as always students should not confer with each other when answering
the sheet.

\paragraph{Before beginning:} for those who have not encountered metric spaces before it may be
a good idea for you to skim-read the Wikipedia article on the subject to familiarise yourself with
key concepts before beginning.\footnote{See \url{https://en.wikipedia.org/wiki/Metric_space}}
However, the exercises have been written in such a way that consulting any such background material
on metric spaces is not strictly necessary, with enough background information embedded within
this document to help you through.
\pagebreak\<close>
  
section\<open>Metric spaces, and some examples\<close>
  
text\<open>As the definition of metric spaces relies crucially on the real numbers, we will be working
with the Isabelle/HOL theory \texttt{Complex\_Main}---rather than the more usual \texttt{Main}---for
this exercise.  This theory imports an implementation of the real and complex numbers, as well as
associated definitions and theorems:\<close>
  
theory Assessed_Exercise_Two_Questions
  imports Complex_Main
begin
  
text\<open>Consider the distance between any two points in $\mathbb{R}^{3}$, the absolute distance between
two integers, or the Lehvenstein (edit) distance between a pair of strings.  Many mathematical
objects have a notion of distance associated with them---and some even have multiple different
reasonable notions of distance. \emph{Metric spaces} are intended as an abstraction of all of these
notions of distance, and of our own informal understanding of this concept.

Formally, a metric space $\langle C, \delta \rangle$ is a carrier set $C$ paired with a
\emph{metric} (or distance) function $\delta : C \times C \rightarrow \mathbb{R}$ that intuitively
represents the distance between any two points of the carrier set, $C$.  We can capture this
structure in Isabelle/HOL using a record with two fields, one corresponding to the carrier, and the
other to the metric function:\<close> 
  
record 'a metric_space =
  carrier :: "'a set"
  metric :: "'a \<Rightarrow> 'a \<Rightarrow> real"
  
text\<open>I note here that, differing slightly from the standard mathematical presentation, I have
Curried the metric function in the Isabelle/HOL modelling of a metric space.  This is a minor
difference that has no impact on the theory.  Also, I use Isabelle's type polymorphism to express
an ambivalence in the type of the contents of the carrier set.

As a supposed generalisation of our intuitive notion of distance, a metric function must
satisfy a number of important laws, or properties, in order for the pair $\langle C, \delta \rangle$
to be a valid metric space.  These are as follows:

  \<^item> It makes no sense to state that one object is a negative distance from some other object:
    distances are only ever zero or positive, and never negative.  Therefore the real number
    returned by a metric space's metric function for any two points must be non-negative.
  \<^item> The distance travelled when walking from Cambridge to St.~Neots is the same as the distance
    travelled when walking from St.~Neots to Cambridge.  Generalising, the distance between a point
    $x$ and a point $y$ must be the same as the distance between the point $y$ and the point $x$.
    That is: a metric space's metric function is \emph{symmetric},
  \<^item> The distance between any two points is zero if and only if the two points are identical.  This
    law is a little harder to justify intuitively, as in natural language we often say that the
    distance between two touching objects is zero.  However, if we think of the elements of the
    carrier set as ``points'' with no internal volume then the axiom asserts that any two co-located
    points are identical,
  \<^item> There are no shortcuts: a metric function must capture the \emph{shortest} distance between
    any two points.  The distance travelled when travelling from point $x$ to $z$ via an
    intermediary point $y$ must be at least as long as the distance travelled simply moving between
    points $x$ and $z$ directly.

The third law above is sometimes known as the ``identity of indiscernibles'', whilst the last law is
sometimes referred to as the ``triangular inequality''.  These laws can be easily captured as a HOL
predicate on \texttt{metric\_space} records, asserting that a carrier set and metric function
pairing constitute a valid metric space:\<close>
  
definition metric_space :: "'a metric_space \<Rightarrow> bool" where
  "metric_space M \<equiv>
     (\<forall>x\<in>carrier M. \<forall>y\<in>carrier M. metric M x y \<ge> 0) \<and>
     (\<forall>x\<in>carrier M. \<forall>y\<in>carrier M. metric M x y = metric M y x) \<and>
     (\<forall>x\<in>carrier M. \<forall>y\<in>carrier M. metric M x y = 0 \<longleftrightarrow> x = y) \<and>
     (\<forall>x\<in>carrier M. \<forall>y\<in>carrier M. \<forall>z\<in>carrier M.
        metric M x z \<le> metric M x y + metric M y z)"
  
text\<open>Note here that I use the same name, \texttt{metric\_space}, to denote both the underlying type
of metric space records as well as the predicate asserting that those records correctly model a
metric space.  This does not matter---the two names live in different namespaces.  Now that we have
a suitable set of definitions for modelling metric spaces in Isabelle/HOL we can begin showing that
some concrete carrier set and metric function pairings are indeed valid metric spaces.  As a first
example, a metric space can be constructed from the set of real numbers by taking the distance
between any two reals, $j$ and $k$, to be their absolute difference, $\mid j - k \mid$.  We can
capture this by defining a suitable instance of the \texttt{metric\_space} record:\<close>
  
definition real_abs_metric_space :: "real metric_space" where
  "real_abs_metric_space \<equiv> \<lparr> carrier = UNIV, metric = \<lambda>x y. abs (x - y) \<rparr>"
  
text\<open>Note here that the carrier set of the metric space is \texttt{UNIV}, the universal set.
Isabelle correctly infers that this has type \texttt{real} \texttt{set} as the type of
\texttt{real\_abs\_metric\_space} has been constrained to the type \texttt{real}
\texttt{metric\_space}.  Now that we have a pairing of a carrier set and metric function, we must
show that this pairing is indeed a valid metric space.  We do this by proving that the
\texttt{metric\_space} predicate holds of this record.

\textbf{Exercise (4 marks)}: prove that the predicate \texttt{metric\_space} holds of
\texttt{real\_abs\_metric\_space} by proving the following theorem.  That is, replace the
\texttt{oops} command below with a complete structured proof.\<close>
  
theorem
  shows "metric_space real_abs_metric_space"
proof(unfold metric_space_def, safe)
  fix x y :: real 
  show "0 \<le> metric real_abs_metric_space x y"
    by (simp add: real_abs_metric_space_def)
next
  fix x y :: real 
  show "metric real_abs_metric_space y x = metric real_abs_metric_space x y"
    by (simp add: real_abs_metric_space_def) 
next
  fix x y :: real
  assume "metric real_abs_metric_space x y = 0"
  thus "x = y"
    by (simp add: real_abs_metric_space_def)
next
  fix y :: real 
  show "metric real_abs_metric_space y y = 0"
    by (simp add: real_abs_metric_space_def) 
next
  fix x y z :: real
  show "metric real_abs_metric_space x z \<le> metric real_abs_metric_space x y + metric real_abs_metric_space y z"
    by (simp add :real_abs_metric_space_def)
qed
  
text\<open>The set of real numbers can be lifted into a metric space in another way, using the so-called
\emph{British Rail metric} which models the tendency of all rail journeys between any two points in
the United Kingdom to proceed by first travelling to London, and then travelling onwards.  (The
French call this the \emph{SNCF metric} due to a similar tendency in Metropolitan France.)  This
metric space can again be captured quite easily in Isabelle/HOL, by following the same pattern as
before.

\textbf{Exercise (4 marks)}: prove that the predicate \texttt{metric\_space} holds of
\texttt{br\_metric\_space} by proving the following theorem.  That is, replace the \texttt{oops}
command below with a complete structured proof.\<close>
  
definition br_metric_space :: "real metric_space" where
  "br_metric_space \<equiv> \<lparr> carrier = UNIV, metric = \<lambda>x y. if x = y then 0 else abs x + abs y \<rparr>"
  
theorem
  shows "metric_space br_metric_space"
proof(unfold metric_space_def, safe)
  fix x y :: real 
  show "0 \<le> metric br_metric_space x y"
    by (simp add: br_metric_space_def)
next
  fix x y :: real 
  show "metric br_metric_space y x = metric br_metric_space x y"
    by (simp add: br_metric_space_def) 
next
  fix x y :: real
  assume assumption: "metric br_metric_space x y = 0"
  {
    assume "x \<noteq> y"
    hence "metric br_metric_space x y \<noteq> 0"
      by (simp add: br_metric_space_def)
    hence False
      using assumption by auto 
  }
  thus "x = y"
    by auto
next
  fix y :: real
  show "metric br_metric_space y y = 0"
    by (simp add: br_metric_space_def) 
next
  fix x y z :: real
  show "metric br_metric_space x z \<le> metric br_metric_space x y + metric br_metric_space y z"
    by (simp add :br_metric_space_def)
qed
    
  
text\<open>As a final example, we consider endowing pairs of integers with a metric.  For pairs of
integers $(i_1, j_1)$ and $(i_2, j_2)$ one can use $\mid i_1 - i_2 \mid + \mid j_1 - j_2 \mid$ as a
metric---sometimes called the taxicab metric.  Proving that this is a valid metric space is a little
more involved than the other two examples, due to fiddling with pairs, but still fairly
straightforward.

\textbf{Exercise (5 marks)}: prove that the predicate \texttt{metric\_space} holds of
\texttt{taxicab\_metric\_space} by proving the following theorem.  That is, replace the \texttt{oops}
command below with a complete structured proof.\<close>

definition taxicab_metric_space :: "(int \<times> int) metric_space" where
  "taxicab_metric_space \<equiv>
    \<lparr> carrier = UNIV, metric = \<lambda>p1 p2. abs (fst p1 - fst p2) + abs (snd p1 - snd p2) \<rparr>"
  
theorem
  shows "metric_space taxicab_metric_space"
proof(unfold metric_space_def, safe, rename_tac [!] a b c d, rename_tac [6] a b c d e f)
  fix a b c d :: int
  show "0 \<le> metric taxicab_metric_space (a, b) (c, d)"
    by (simp add: taxicab_metric_space_def )
next
  fix a b c d :: int
  show "metric taxicab_metric_space (a, b) (c, d) = metric taxicab_metric_space (c, d) (a, b)"
    by (simp add: taxicab_metric_space_def )
next
  fix a b c d :: int
  assume "metric taxicab_metric_space (a, b) (c, d) = 0"
  thus "a = c"
    by (simp add: taxicab_metric_space_def )
next
  fix a b c d :: int
  assume "metric taxicab_metric_space (a, b) (c, d) = 0"
  thus "b = d"
    by (simp add: taxicab_metric_space_def)
next
  fix c d :: int
  show "metric taxicab_metric_space (c, d) (c, d) = 0"
    by (simp add: taxicab_metric_space_def )
next
  fix a b c d e f :: int
  show "metric taxicab_metric_space (a, b) (e, f) \<le> metric taxicab_metric_space (a, b) (c, d) + metric taxicab_metric_space (c, d) (e, f)"
    by (simp add: taxicab_metric_space_def )
qed
  
  
section\<open>Making new metric spaces from old\<close>
  
text\<open>We now have a handful of concrete metric spaces.  Given such a collection of existing metric
spaces, can we produce new metric spaces using generic constructions?  That is, are there operations
that take an arbitrary metric space and can produce new ones?  In this section, we explore three
different ways of building new metric spaces from old: restricting a metric space to a subset of the
carrier, shifting a metric via a non-zero constant, and finally taking the product of two metric
spaces.  The first two are relatively straightforward:

\textbf{Exercise (3 marks)}: Suppose $\langle C, \delta\rangle$ is a metric space and suppose
$S \subseteq C$.  Show that $\langle S, \delta \rangle$ is also a metric space by stating and
proving (with a structured proof) a relevant lemma.
  
\textbf{Exercise (3 marks)}: Suppose $\langle C, \delta \rangle$ is a metric space.  Suppose also
that $k>0$ and that $\delta' (x, y) = k \cdot \delta (x, y)$.  Show that
$\langle C, \delta' \rangle$ is a metric space by stating and proving (with a structured proof) a
relevant lemma.

Lastly, we consider taking the product of two metric spaces.  Recall that mathematically
$S \times T$ denotes the \emph{Cartesian product} of two sets, consisting of all ordered pairs
$(s, t)$ where $s \in S$ and $t \in T$.

\textbf{Exercise (5 marks)}: Suppose $\langle S, \delta_1\rangle$ and $\langle T, \delta_2\rangle$
are metric spaces.  Show that the set $S \times T$ can be lifted into a metric space by first
finding a suitable metric and thereafter proving (with a strucured proof) a relevant lemma.  Your
metric on $S \times T$ must make use of both $\delta_1$ and $\delta_2$.\<close>
  
lemma subset_metric_space:
  assumes "metric_space \<lparr>carrier = C, metric = \<delta>\<rparr>" and
          "S \<subseteq> C"
        shows "metric_space \<lparr>carrier = S, metric = \<delta>\<rparr>"
proof(unfold metric_space_def, clarsimp, safe)
  fix x y
  assume "x \<in> S" and "y \<in> S" 
  hence 1: "x \<in> C" and 2:"y \<in> C"
    using assms by auto
  thus "0 \<le> \<delta> x y" and "\<delta> x y = \<delta> y x" 
    using assms by (auto simp add: metric_space_def) 

  assume "\<delta> x y = 0"
  hence "x = y"
    using 1 and 2 and assms by (auto simp add: metric_space_def) 
  
  sorry
    
    
  
  
  
         
    
section\<open>Continuous functions, and some examples\<close>
  
text\<open>One reason why metric spaces are mathematically interesting is because they provide an abstract
venue within which one can define the important notion of \emph{continuous function}, a core concept
in topology and analysis.  Indeed, metric spaces can be seen as a precursor to topology.

Suppose that $\langle S, \delta_1\rangle$ and $\langle T, \delta_2\rangle$ are metric spaces, and
$f : S \rightarrow T$ is a function mapping elements of $S$ to $T$.  Suppose also that $s \in S$ is
a point in $S$.  We say that the function $f$ is \emph{continuous at the point $s$} if for every
$s' \in S$ and $\epsilon>0$ if there exists $d>0$ such that $\delta_1 (s', s) < d$ then
$\delta_2 (f x, f s) < \epsilon$.  Further, call the function $f : S \rightarrow T$
\emph{continuous} if $f$ is continuous at every point $s \in S$.  These two definitions can be
captured in Isabelle/HOL as follows:\<close>
  
context fixes M1 :: "'a metric_space" and M2 :: "'b metric_space" begin
  
definition continuous_at :: "('a \<Rightarrow> 'b) \<Rightarrow> 'a \<Rightarrow> bool" where
  "continuous_at f a \<equiv> \<forall>x\<in>carrier M1. \<forall>\<epsilon>>0.
    (\<exists>d>0. metric M1 x a < d \<longrightarrow> metric M2 (f x) (f a) < \<epsilon>)"
  
definition continuous :: "('a \<Rightarrow> 'b) \<Rightarrow> bool" where
  "continuous f \<equiv> \<forall>x\<in>carrier M1. continuous_at f x"
  
end
  
text\<open>As an aside, here I use a \texttt{context} block to fix two arbitrary metric
spaces---\texttt{M1} and \texttt{M2}---of the correct type for the duration of my definitions.  This
means that I do not need to add the two metric spaces as explicit parameters to the
\texttt{continuous\_at} and \texttt{continuous} functions but can declare them as parameters ``up
front''.  Inspect the type of the definitions to see what \texttt{context} does:\<close>
  
term continuous_at
term continuous

text\<open>Which functions are continuous?  One obvious contender is the identity function.  Suppose
$\langle S, \delta\rangle$ is a metric space.  Then the identity function $id : S \rightarrow S$
maps elements of the carrier $S$ onto elements of the carrier $S$---that is, the identity function
can be seen as a map from a metric space back onto itself.

\textbf{Exercise (3 marks)}: show that the identity function $id$ is a continuous function between
a metric space and itself by stating and proving (with a strucured proof) a relevant lemma.

Constant functions are also continuous.  Suppose $\langle S, \delta_1\rangle$ and
$\langle T, \delta_2\rangle$ are metric spaces and $t \in T$ is a point in $T$.  Suppose also that
$f : S \rightarrow T$ maps all $s$ to $t$ (i.e. it is a constant function that always returns
$t$).  Then $f$ is continuous.

\textbf{Exercise (4 marks)}: show that constant functions are continuous by proving the following
lemma.  That is, replace the \texttt{oops} command below with a complete structured proof.\<close>

lemma continuous_const:
  assumes "metric_space M1" and "metric_space M2"
    and "y \<in> carrier M2"
  shows "continuous M1 M2 (\<lambda>x. y)"
  oops
    
text\<open>Lastly, suppose $\langle S, \delta_1\rangle$, $\langle T, \delta_2\rangle$, and
$\langle U, \delta_3\rangle$ are metric spaces.  Suppose also that $f : S \rightarrow T$ and
$g : T \rightarrow U$ are continuous functions between relevant metric spaces.  Then, providing
that for every $s \in S$ we have $f s \in T$ holds, the composition $(g \circ f) : S \rightarrow U$
is also a continuous function between the metric spaces $\langle S, \delta_1\rangle$ and
$\langle U, \delta_3\rangle$.

\textbf{Exercise (6 marks)}: show that the composition of two continuous functions is continuous by
proving the following lemma.  That is, replace the \texttt{oops} command below with a complete
structured proof.\<close>
  
lemma continuous_comp:
  assumes "metric_space M1" and "metric_space M2" and "metric_space M3"
    and "continuous M1 M2 f" and "continuous M2 M3 g"
    and "\<And>x. x \<in> carrier M1 \<Longrightarrow> f x \<in> carrier M2"
  shows "continuous M1 M3 (g o f)"
oops
  
section\<open>Open balls\<close>
  
text\<open>Suppose $\langle S, \delta\rangle$ is a metric space and $c \in S$ is a point in $S$.  Suppose
also that $r>0$ is some strictly positive real number.  Define the \emph{open ball of radius $r$
around the point $c$} as the set of all points in $S$ that are strictly less than $r$ distance away
from the point $c$ when measured using the metric $\delta$.  Where the underlying metric space is
obvious, I will write $\mathcal{B}(c,r)$ for the open ball around point $c$ of radius $r$.

\textbf{Exercise (2 marks)}: Suppose $\langle S, \delta\rangle$ is a metric space.  Define the open
ball $\mathcal{B}(c,r)$ in this metric space by completing the definition of \texttt{open\_ball}.
That is, replace the \texttt{consts} declaration below with a complete definition.\<close>

consts open_ball :: "'a metric_space \<Rightarrow> 'a \<Rightarrow> real \<Rightarrow> 'a set"
   
text\<open>For any open ball $\mathcal{B}(c,r)$ in a metric space $\langle S, \delta\rangle$ we have that
$\mathcal{B}(c,r) \subseteq S$, i.e. open balls are always subsets of the underlying metric space's
carrier set.  This fact holds for any ball of any radius.

\textbf{Exercise (2 marks)}: show that an arbitrary open ball in a fixed metric space is a subset of
that metric space's carrier set by stating and proving (with a structured proof) a relevant lemma.

Moreover, we have that in a fixed metric space, the open ball $\mathcal{B}(c,0) = \{\}$.  That is,
open balls of zero radius are the empty set.

\textbf{Exercise (2 marks)}: show that an open ball in a fixed metric space with radius $0$ is the
empty set.

Additionally, it should be intuitively obvious that for any open ball in a fixed metric space
$\mathcal{B}(c, r)$ where $r > 0$ we have the property that $c \in \mathcal{B}(c, r)$, i.e. the open
ball contains its own centre as a point.

\textbf{Exercise (3 marks)}: show that any open ball in a fixed metric space with strictly positive
radius contains its centre as a point by proving the following lemma.  That is, replace the
\texttt{oops} command below with a complete structured proof.\<close>
    
lemma centre_in_open_ball:
  assumes "metric_space M" and "c \<in> carrier M"
    and "0 < r"
  shows "c \<in> open_ball M c r"
  oops
    
text\<open>Lastly, suppose we have two open balls around the same centre point---$\mathcal{B}(c, r)$ and
$\mathcal{B}(c, s)$---such that $r \leq s$ where $c$ is contained within some ambient fixed metric
space.  Then it should be obvious that the open ball with smaller radius is a subset of the open
ball with the larger radius.

\textbf{Exercise (4 marks)}: show that an open ball around a fixed centre point with a smaller
radius than another open ball around the same fixed centre point is a subset of the latter open ball
by proving the following theorem.  That is, replace the \texttt{oops} command below with a complete
structured proof.\<close>
  
lemma open_ball_le_subset:
  assumes "metric_space M"
    and "c \<in> carrier M" and "r \<le> s"
  shows "open_ball M c r \<subseteq> open_ball M c s"
  oops
  
end
  
text\<open>
\begin{center}
\emph{The end\ldots}
\end{center}\<close>
