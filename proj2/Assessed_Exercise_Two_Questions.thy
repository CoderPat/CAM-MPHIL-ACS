
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
  
  
locale metric_space =
  fixes carrier :: "'a set"
    and metric :: "'a \<Rightarrow> 'a \<Rightarrow> real"
  assumes non_negative_metric:"(\<forall>x\<in>carrier . \<forall>y\<in>carrier . metric x y \<ge> 0)"
     and  reflexive_metric:   "(\<forall>x\<in>carrier . \<forall>y\<in>carrier . metric x y = metric y x)"
     and  discernible_metric: "(\<forall>x\<in>carrier . \<forall>y\<in>carrier . metric x y = 0 \<longleftrightarrow> x = y)"
     and  subadditive_metric: "(\<forall>x\<in>carrier . \<forall>y\<in>carrier . \<forall>z\<in>carrier.
                                  metric x z \<le> metric x y + metric y z)"
  
text\<open>Note here that I use the same name, \texttt{metric\_space}, to denote both the underlying type
of metric space records as well as the predicate asserting that those records correctly model a
metric space.  This does not matter---the two names live in different namespaces.  Now that we have
a suitable set of definitions for modelling metric spaces in Isabelle/HOL we can begin showing that
some concrete carrier set and metric function pairings are indeed valid metric spaces.  As a first
example, a metric space can be constructed from the set of real numbers by taking the distance
between any two reals, $j$ and $k$, to be their absolute difference, $\mid j - k \mid$.  We can
capture this by defining a suitable instance of the \texttt{metric\_space} record:\<close>
  
  
text\<open>Note here that the carrier set of the metric space is \texttt{UNIV}, the universal set.
Isabelle correctly infers that this has type \texttt{real} \texttt{set} as the type of
\texttt{real\_abs\_metric\_space} has been constrained to the type \texttt{real}
\texttt{metric\_space}.  Now that we have a pairing of a carrier set and metric function, we must
show that this pairing is indeed a valid metric space.  We do this by proving that the
\texttt{metric\_space} predicate holds of this record.

\textbf{Exercise (4 marks)}: prove that the predicate \texttt{metric\_space} holds of
\texttt{real\_abs\_metric\_space} by proving the following theorem.  That is, replace the
\texttt{oops} command below with a complete structured proof.\<close>

(* 4 marks *)   
interpretation real_metric_space : metric_space "UNIV" "\<lambda>(x::real) (y::real). abs (x - y)"
proof(standard, safe)
  fix x y :: real
  show "0 \<le> \<bar>x - y\<bar>" and "\<bar>x - y\<bar> = \<bar>y - x\<bar>"
    by (auto)
  
  assume "\<bar>x - y\<bar> = 0"
  thus "x = y"
    by auto
next
  fix y :: real
  show "\<bar>y - y\<bar> = 0"
    by auto
next
  fix x y z :: real
  show "\<bar>x - z\<bar> \<le> \<bar>x - y\<bar> + \<bar>y - z\<bar>"
    by auto
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
  
(* 4 marks *)  
interpretation br_metric_space : metric_space UNIV
                                     "\<lambda> (x::real) (y::real). if x = y then 0 else abs x + abs y"
proof(standard, safe)
  fix x y :: real
  show "0 \<le> (if x = y then 0 else \<bar>x\<bar> + \<bar>y\<bar>)"
    by auto
  show "(if x = y then 0 else \<bar>x\<bar> + \<bar>y\<bar>) = (if y = x then 0 else \<bar>y\<bar> + \<bar>x\<bar>)"
    by auto
   
  assume assm: "(if x = y then 0 else \<bar>x\<bar> + \<bar>y\<bar>) = 0"
  {
    assume "x\<noteq>y"
    hence "(if x = y then 0 else \<bar>x\<bar> + \<bar>y\<bar>) \<noteq> 0"
      by auto
    hence False
      using assm by auto
  }
  thus "x = y"
    by auto
next
  fix y ::real
  show "(if y = y then 0 else \<bar>y\<bar> + \<bar>y\<bar>) = 0"
    by auto
next
  fix x y z :: real
  show "(if x = z then 0 else \<bar>x\<bar> + \<bar>z\<bar>) \<le> 
          (if x = y then 0 else \<bar>x\<bar> + \<bar>y\<bar>) + (if y = z then 0 else \<bar>y\<bar> + \<bar>z\<bar>)"
    by auto
qed
  
  
text\<open>As a final example, we consider endowing pairs of integers with a metric.  For pairs of
integers $(i_1, j_1)$ and $(i_2, j_2)$ one can use $\mid i_1 - i_2 \mid + \mid j_1 - j_2 \mid$ as a
metric---sometimes called the taxicab metric.  Proving that this is a valid metric space is a little
more involved than the other two examples, due to fiddling with pairs, but still fairly
straightforward.

\textbf{Exercise (5 marks)}: prove that the predicate \texttt{metric\_space} holds of
\texttt{taxicab\_metric\_space} by proving the following theorem.  That is, replace the \texttt{oops}
command below with a complete structured proof.\<close>

(* 5 marks *)
interpretation taxicab_metric_space : metric_space "UNIV" 
                           "\<lambda>(x1::int, x2::int) (y1::int, y2::int). abs (x1 - y1) + abs (x2 - y2)"
proof(standard, safe)
  fix x1 x2 y1 y2 :: int
  show "0 \<le> real_of_int(\<bar>x1 - y1\<bar> + \<bar>x2 - y2\<bar>)"
    by simp
  show "real_of_int (\<bar>x1 - y1\<bar> + \<bar>x2 - y2\<bar>) = real_of_int (\<bar>y1 - x1\<bar> + \<bar>y2 -x2\<bar>)"
    by simp
  
  assume a:"real_of_int (\<bar>x1 - y1\<bar> + \<bar>x2 - y2\<bar>)  = 0"
  {
    assume 1:"\<bar>x1 - y1\<bar> \<noteq> 0 \<or> \<bar>x2 - y2\<bar> \<noteq> 0"
    have "\<bar>x1 - y1\<bar> \<ge> 0 \<and> \<bar>x2 - y2\<bar> \<ge> 0"
      by auto
    hence "\<bar>x1 - y1\<bar> > 0 \<and> \<bar>x2 - y2\<bar> > 0"
      using 1 a by linarith
    hence "real_of_int (\<bar>x1 - y1\<bar> + \<bar>x2 - y2\<bar>) > 0"
      by auto
    hence False
      using a by auto
  }
  thus "x1 = y1" and "x2 = y2"
    by auto
next
  fix x1 x2
  show "real_of_int (\<bar>x1 - x1\<bar> + \<bar>x2 - x2\<bar>) = 0"
    by auto
next
  fix x1 x2 y1 y2 z1 z2
  show "real_of_int (\<bar>x1 - z1\<bar> + \<bar>x2 - z2\<bar>) \<le> 
              real_of_int (\<bar>x1 - y1\<bar> + \<bar>x2 - y2\<bar>) + real_of_int (\<bar>y1 - z1\<bar> + \<bar>y2 - z2\<bar>)"
    by linarith
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

(* 3 marks *)

lemma subset_metric_space:
  assumes "metric_space C \<delta>" and
          "S \<subseteq> C"
        shows "metric_space S \<delta>"
proof(unfold metric_space_def, clarsimp, safe)
  fix x y
  assume "x \<in> S" and "y \<in> S" 
  hence 1: "x \<in> C" "y \<in> C"
    using assms by auto
  thus "0 \<le> \<delta> x y" and "\<delta> x y = \<delta> y x" 
    using assms by (auto simp add: metric_space_def) 

  assume "\<delta> x y = 0"
  thus "x = y"
    using 1 and assms by (simp add: metric_space_def) 
next
  fix y
  assume "y \<in> S" 
  hence "y \<in> C"
    using assms by auto
  thus "\<delta> y y = 0"
    using assms by (simp add:metric_space_def)
next
  fix x y z
  assume "x \<in> S" "y \<in> S" "z \<in> S"
  hence "x \<in> C" "y \<in> C" "z \<in> C"
    using assms by auto
  thus "\<delta> x z \<le> \<delta> x y + \<delta> y z"
    using assms by (simp add:metric_space_def)
qed

(* 3 marks *)
lemma scale_metric:
  assumes "metric_space S \<delta>" 
    and "\<omega> = (\<lambda>x1 x2. k * (\<delta> x1 x2)) " and "k>0"
  shows "metric_space S \<omega>"
proof(unfold metric_space_def, clarsimp, safe)
  fix x y
  assume 1:"x \<in> S" "y \<in> S"
  hence "\<delta> x y \<ge> 0"
    using assms by (auto simp add: metric_space_def)
  thus "\<omega> x y \<ge> 0"
    using assms by (auto)
  
  have "\<delta> x y = \<delta> y x"
    using 1 and assms by (auto simp add: metric_space_def)
  thus "\<omega> x y = \<omega> y x"
    using assms by (auto)
  
  assume "\<omega> x y = 0"
  hence "\<delta> x y = 0"
    using assms by auto
  thus "x = y"
    using 1 and assms by (auto simp add: metric_space_def)
next
  fix y
  assume "y \<in> S"
  hence "\<delta> y y = 0"
    using assms by (auto simp add: metric_space_def)
  thus "\<omega> y y = 0"
    using assms by auto
next
  fix x y z
  assume "x \<in> S" and "y \<in> S" and  "z \<in> S"
  hence "\<delta> x z \<le> \<delta> x y + \<delta> y z"
    using assms by (auto simp add: metric_space_def)
  hence "k * (\<delta> x z) \<le> k*(\<delta> x y + \<delta> y z)"
    using assms by auto
  hence "k * (\<delta> x z) \<le> k*(\<delta> x y) + k*( \<delta> y z)"
    using assms  by (simp add: ring_distribs)
  thus "\<omega> x z \<le> \<omega> x y + \<omega> y z"
    using assms by auto
qed
  
(* 5 marks *)
lemma product_metric_spaces:
  assumes "metric_space C1 \<delta>1" 
     and "metric_space C2 \<delta>2"
     and "\<omega> = (\<lambda>(x1, x2) (y1, y2). (\<delta>1 x1 y1) + (\<delta>2 x2 y2))"
   shows "metric_space (C1\<times>C2) \<omega>"
proof(unfold metric_space_def, clarsimp, safe)
  fix x1 y1 x2 y2
  assume 1:"x1 \<in> C1" "y1 \<in> C1" "x2 \<in> C2" "y2 \<in> C2"
  hence "\<delta>1 x1 y1 \<ge> 0" and "\<delta>2 x2 y2 \<ge> 0"
    using assms by (auto simp add: metric_space_def)
  thus "\<omega> (x1, x2) (y1, y2) \<ge> 0"
    using assms by auto
  
  have "\<delta>1 x1 y1 = \<delta>1 y1 x1" and "\<delta>2 x2 y2 = \<delta>2 y2 x2"
    using 1 and assms  by (auto simp add: metric_space_def)
  thus "\<omega> (x1, x2) (y1, y2) = \<omega> (y1, y2) (x1, x2)"
    using assms by auto
      
  assume 2:"\<omega> (x1, x2) (y1, y2) = 0"
  {
    assume 3:"\<delta>1 x1 y1 \<noteq> 0 \<or> \<delta>2 x2 y2 \<noteq> 0"
    have 4: "\<delta>1 x1 y1 \<ge> 0 \<and> \<delta>2 x2 y2 \<ge> 0"
      using 1  and assms by (auto simp add: metric_space_def)
    hence "\<delta>1 x1 y1 > 0 \<or> \<delta>2 x2 y2 > 0"
      using 3 by auto
    hence "\<omega> (x1, x2) (y1, y2) > 0"
      using 4 and assms by auto
    hence False
      using 2 by auto
  }
  hence "\<delta>1 x1 y1 = 0" and "\<delta>2 x2 y2 = 0"
    by auto
  thus "x1 = y1" and "x2 = y2"
    using 1 and assms by (auto simp add: metric_space_def)
next
  fix x1 x2
  assume "x1 \<in> C1" "x2 \<in> C2"
  hence "\<delta>1 x1 x1 = 0" and "\<delta>2 x2 x2 = 0"
    using assms by (auto simp add: metric_space_def)
  thus "\<omega> (x1, x2) (x1, x2) = 0"
    using assms by auto
next
  fix x1 x2 y1 y2 z1 z2
  assume "x1 \<in> C1" "y1 \<in> C1" "z1 \<in> C1" "x2 \<in> C2" "y2 \<in> C2" "z2 \<in> C2"
  hence "\<delta>1 x1 z1 \<le> \<delta>1 x1 y1 + \<delta>1 y1 z1" and "\<delta>2 x2 z2 \<le> \<delta>2 x2 y2 + \<delta>2 y2 z2"
    using assms by (auto simp add: metric_space_def)
  thus "\<omega> (x1, x2) (z1, z2) \<le> \<omega> (x1, x2) (y1, y2) + \<omega> (y1, y2) (z1, z2)"
    using assms by auto
qed
  

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
  

context fixes domain :: "'a set" and metric_d :: "'a \<Rightarrow> 'a \<Rightarrow> real"
       and  codomain :: "'b set" and metric_c :: "'b \<Rightarrow> 'b \<Rightarrow> real"
begin   
  
definition  continuous_at :: "('a \<Rightarrow> 'b) \<Rightarrow> 'a \<Rightarrow> bool" where
  "continuous_at f a \<equiv> \<forall>x\<in>domain. \<forall>\<epsilon>>0.
    (\<exists>d>0. metric_d x a < d \<longrightarrow> metric_c  (f x) (f a) < \<epsilon>)"
  
definition  continuous :: "('a \<Rightarrow> 'b) \<Rightarrow> bool" where
  "continuous f \<equiv> \<forall>x\<in>domain. continuous_at f x"

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

(* 3 marks *) 
lemma continuous_id:
  assumes "metric_space domain metric_d"
  shows "continuous domain metric_d metric_d (\<lambda>x. x)"
proof(simp add:continuous_def continuous_at_def, safe)
  fix x1 x2
  fix \<epsilon> :: real
  assume 1: "0 < \<epsilon>"
  {
    assume "metric_d x2 x1 < \<epsilon>"
    hence "metric_d x2 x1 < \<epsilon>"
      by auto
  }
  thus "\<exists>d>0. metric_d x2 x1 < d \<longrightarrow> metric_d x2 x1 < \<epsilon>"
    using 1 by auto
qed

(*4 marks *) 
lemma continuous_const:
  assumes "metric_space domain metric_d" and "metric_space codomain metric_c"
    and "y \<in> codomain"
  shows "continuous domain metric_d metric_c (\<lambda>x. y)"
proof(simp add:continuous_def continuous_at_def, safe)
  fix x1 x2
  fix \<epsilon> :: real
  assume 1: "0 < \<epsilon>"
  {
    assume "metric_d x2 x1 < \<epsilon>"
    have "metric_c y y = 0"
      using assms by (auto simp add: metric_space_def)
    hence "metric_c y y < \<epsilon>"
      using 1 by auto
  }
  thus "\<exists>d>0. metric_d x2 x1 < d \<longrightarrow> metric_c y y < \<epsilon>"
    using 1 by auto
qed
   
  
text\<open>Lastly, suppose $\langle S, \delta_1\rangle$, $\langle T, \delta_2\rangle$, and
$\langle U, \delta_3\rangle$ are metric spaces.  Suppose also that $f : S \rightarrow T$ and
$g : T \rightarrow U$ are continuous functions between relevant metric spaces.  Then, providing
that for every $s \in S$ we have $f s \in T$ holds, the composition $(g \circ f) : S \rightarrow U$
is also a continuous function between the metric spaces $\langle S, \delta_1\rangle$ and
$\langle U, \delta_3\rangle$.

\textbf{Exercise (6 marks)}: show that the composition of two continuous functions is continuous by
proving the following lemma.  That is, replace the \texttt{oops} command below with a complete
structured proof.\<close>

(* 6 marks *) 
lemma continuous_comp:
  assumes "metric_space domain metric_d" and "metric_space inters metric_i"
    and "metric_space codomain metric_c"
    and "continuous domain metric_d metric_i f"
    and "continuous inters metric_i metric_c g"
    and "\<And>x. x \<in> domain \<Longrightarrow> f x \<in> inters"
  shows "continuous domain metric_d metric_c (g o f)"
proof(simp add:continuous_def continuous_at_def, safe)
  fix x1 x2
  fix \<epsilon> :: real
  assume 1:"\<epsilon> > 0"
  assume 2:"x1 \<in> domain" and 3:"x2 \<in> domain"  
  hence "(f x1) \<in> inters" and "(f x2) \<in> inters"
    using assms by auto
  hence "\<exists>d>0. metric_i (f x2) (f x1) < d \<longrightarrow> metric_c (g (f x2)) (g (f x1)) < \<epsilon>"
    using 1 and assms by (auto simp add:continuous_def continuous_at_def)
  then obtain k::real where 4:"k>0" and 5:"metric_i (f x2) (f x1) < k \<longrightarrow> metric_c (g (f x2)) (g (f x1)) < \<epsilon>"
    by auto
      
  have "\<exists>d>0. metric_d x2 x1 < d \<longrightarrow> metric_i (f x2) (f x1) < k"
    using 2 3 4 and assms by (auto simp add:continuous_def continuous_at_def)
  then obtain d::real where 6:"d>0" and 7:"metric_d x2 x1 < d \<longrightarrow> metric_i (f x2) (f x1) < k"
    by auto
  {
    assume "metric_d x2 x1 < d"
    hence "metric_c (g (f x2)) (g (f x1)) < \<epsilon>"
      using  5 and 7 by (auto)
  }
  thus "\<exists>d>0. metric_d x2 x1 < d \<longrightarrow> metric_c (g (f x2)) (g (f x1)) < \<epsilon>"
    using 6 by auto
qed
  
section\<open>Open balls\<close>
  
text\<open>Suppose $\langle S, \delta\rangle$ is a metric space and $c \in S$ is a point in $S$.  Suppose
also that $r>0$ is some strictly positive real number.  Define the \emph{open ball of radius $r$
around the point $c$} as the set of all points in $S$ that are strictly less than $r$ distance away
from the point $c$ when measured using the metric $\delta$.  Where the underlying metric space is
obvious, I will write $\mathcal{B}(c,r)$ for the open ball around point $c$ of radius $r$.

\textbf{Exercise (2 marks)}: Suppose $\langle S, \delta\rangle$ is a metric space.  Define the open
ball $\mathcal{B}(c,r)$ in this metric space by completing the definition of \texttt{open\_ball}.
That is, replace the \texttt{consts} declaration below with a complete definition.\<close>

(* 2 marks *) 
  
context metric_space begin
  
definition open_ball :: "'a \<Rightarrow> real \<Rightarrow> 'a set" where
  "open_ball c r \<equiv> { x \<in> carrier. metric c x < r }"
    
end
  
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

(* 2 marks *) 
  
context metric_space begin

lemma open_ball_subset_carrier:
  assumes "c \<in> carrier"
  shows "open_ball c r \<subseteq> carrier"
proof -
  show "open_ball c r \<subseteq> carrier"
    using assms by(auto simp add: open_ball_def)
qed
  
(* 2 marks *)  
lemma empty_ball:
  assumes "c \<in> carrier"
  shows "open_ball c 0 = {}"
proof -
  {
    fix x
    assume 1:"x \<in> open_ball c 0"
    hence 2: "x \<in> carrier"
      by (auto simp add: open_ball_def)
    have 3: "metric c x < 0"
      using 1 by (auto simp add: open_ball_def)
    have "metric c x \<ge> 0"
       using 2 and assms by (simp add: non_negative_metric)
    hence False 
      using 3 by auto
  }
  thus "open_ball c 0 = {}"
    by auto
qed
    
(* 3 marks *)
lemma centre_in_open_ball:
  assumes "c \<in> carrier" and "0 < r"
  shows "c \<in> open_ball c r"
proof -
  have "metric c c = 0"
    using assms by (simp add: discernible_metric)
  hence "metric c c < r" 
    using assms by auto
  thus "c \<in> open_ball c r"
    using assms by (auto simp add: open_ball_def)
qed
  
    
text\<open>Lastly, suppose we have two open balls around the same centre point---$\mathcal{B}(c, r)$ and
$\mathcal{B}(c, s)$---such that $r \leq s$ where $c$ is contained within some ambient fixed metric
space.  Then it should be obvious that the open ball with smaller radius is a subset of the open
ball with the larger radius.

\textbf{Exercise (4 marks)}: show that an open ball around a fixed centre point with a smaller
radius than another open ball around the same fixed centre point is a subset of the latter open ball
by proving the following theorem.  That is, replace the \texttt{oops} command below with a complete
structured proof.\<close>
  
(* 4 marks *)
lemma open_ball_le_subset:
  assumes "c \<in> carrier" and "r \<le> s"
  shows "open_ball c r \<subseteq> open_ball c s"
proof
  fix x
  assume "x \<in> open_ball c r"
  hence "metric c x < r" and 1:"x \<in> carrier"
    by (auto simp add: open_ball_def)
  hence "metric c x < s"
    using assms by simp
  thus "x \<in> open_ball c s"
    using 1 and assms by (simp add: open_ball_def)
qed

end 
  
section\<open>Cauchy and Convergence\<close>
  
context metric_space
begin

definition cauchy where
  "cauchy seq \<equiv> (\<forall>\<epsilon>>0. \<exists>p. \<forall>m\<ge>p. \<forall>n\<ge>p . m>n \<longrightarrow> metric (seq m) (seq n) < \<epsilon>)"

definition is_limit where
  "is_limit seq l \<equiv> \<forall>\<epsilon>>0. \<exists>p. \<forall>n\<ge>p. metric (seq n) l < \<epsilon>"
  
lemma unique_limit:
  fixes seq :: "nat \<Rightarrow> 'a"
  assumes "\<forall>n. seq n \<in> carrier" 
  assumes "x \<in> carrier" and "y \<in> carrier"
  assumes "is_limit seq x" and "is_limit seq y"
  shows "x = y"
proof -
  {
  assume "x \<noteq> y"
  hence pos:"metric x y > 0"
    using assms discernible_metric less_eq_real_def non_negative_metric by auto
  obtain p1 where p1:"\<forall>n\<ge>p1. metric (seq n) x < metric x y / 2"
    using assms pos half_gt_zero is_limit_def by blast
  obtain p2 where p2:"\<forall>n\<ge>p2. metric (seq n) y < metric x y / 2"
    using assms pos half_gt_zero is_limit_def by blast
  obtain n where "n\<ge>p1" and "n\<ge>p2"
  proof
    show "p1 + p2 \<ge> p1"
      by simp
    show "p1 + p2 \<ge> p2"
      by simp
  qed
  hence "metric (seq n) x < metric x y / 2" and "metric (seq n) y < metric x y / 2"
    using p1 p2 by auto
  hence 1:"metric (seq n) x + metric (seq n) y < metric x y "
    by auto
  have "metric x y \<le> metric (seq n) x + metric (seq n) y"
    using assms reflexive_metric subadditive_metric by auto
  hence "metric x y < metric x y"
    using "1" by linarith
  hence "False"
    by auto
  }
  thus "x = y"
    by auto
qed

thm continuous_at_def  
lemma continuous_limits:
  fixes seq :: "nat \<Rightarrow> 'a"
  assumes "\<And>x. x \<in> carrier \<Longrightarrow> f x \<in> carrier"
  assumes "\<forall>n. seq n \<in> carrier" "l \<in> carrier"
  assumes "continuous_at carrier metric metric_c f l" 
  assumes "is_limit seq l"
  shows "is_limit (f \<circ> seq) (f l)"
proof(unfold is_limit_def, unfold o_def, standard)
  fix n :: nat
  have "seq n \<in> carrier"
    using assms by auto
  hence 1:"\<forall>\<epsilon>>0. \<exists>d>0. metric (seq n) l < d \<longrightarrow> metric_c (f (seq n)) (f l) < \<epsilon>"
    by (meson assms(4) continuous_at_def)
  fix \<epsilon> :: real
  assume \<epsilon>:"\<epsilon> > 0"
  then obtain d where d:"d>0" "metric (seq n) l < d \<longrightarrow> metric_c (f (seq n)) (f l) < \<epsilon>"
    using "1" by blast
  hence "\<exists>p. \<forall>n\<ge>p. metric (seq n) l < d"
    using assms(5) is_limit_def by auto
  then obtain p where "\<forall>n\<ge>p. metric (seq n) l < d"
    by auto
  
  
  

lemma limits_preserved:
  shows "is_limit seq l = is_limit (\<lambda>n. seq (n+1)) l"
  sorry

definition convergent where
  "convergent seq \<equiv> (\<exists>x0\<in>carrier. is_limit seq x0)"
   
lemma convergent_cauchy:
  fixes seq::"nat \<Rightarrow> 'a" 
  assumes "\<forall>n. seq n \<in> carrier" and "convergent seq"
  shows "cauchy seq"
proof(unfold cauchy_def, safe)
  fix \<epsilon> :: real
  assume \<epsilon>:"\<epsilon> > 0"
  obtain \<delta>::real where \<delta>:"\<epsilon> = \<delta>*2" "\<delta> > 0"
  proof
    show "\<epsilon> = (\<epsilon>/2) * 2"
      by auto
    show "\<epsilon>/2 > 0"
      using \<epsilon> by auto
  qed
  obtain x0 where 2:"x0 \<in> carrier" and "\<forall>\<epsilon>>0. \<exists>p. \<forall>n\<ge>p. metric (seq n) x0 < \<epsilon>"
    using assms by (auto simp add: convergent_def is_limit_def)
  hence "\<exists>p. \<forall>n\<ge>p. metric (seq n) x0 < \<delta>"
    using \<delta> by auto
  then obtain p where 3:"\<forall>n\<ge>p. metric (seq n) x0 < \<delta>"
    by auto
  {
    fix n m :: nat
    assume "m \<ge> p" and "n \<ge> p"
    hence  "metric (seq n) x0 < \<delta>" and "metric (seq m) x0 < \<delta>"
      using 3 by auto
    hence 4:"metric (seq n) x0 < \<delta>" and 5:"metric x0 (seq m) < \<delta>"
      using 2 and assms by (auto simp add: reflexive_metric)
    have "metric (seq n) (seq m) \<le> metric (seq n) x0 + metric x0 (seq m)"
      using 2 and assms by (auto simp add:subadditive_metric)
    hence "metric (seq n) (seq m) < \<delta> + \<delta>"
      using 4 5 by auto
    hence "metric (seq n) (seq m) < \<epsilon>"
      using \<delta> by auto
  }
  thus "\<exists>p. \<forall>m\<ge>p. \<forall>n\<ge>p. m>n \<longrightarrow> metric (seq m) (seq n) < \<epsilon>"
    using \<epsilon> by auto
qed
end

section\<open>Complete Metric Spaces and Banach Fixed Point Theorem\<close>
  
locale complete_metric_space = metric_space +
  assumes completness: "\<forall>seq::(nat\<Rightarrow>'a). cauchy seq \<longrightarrow> convergent seq"

context complete_metric_space
begin
  
definition contraction_map where
  "contraction_map f \<equiv> (\<exists>q\<ge>0. q<1 \<and> (\<forall>x\<in>carrier. \<forall>y\<in>carrier. metric (f x) (f y) \<le> q * metric x y))"
  
fun iter ::  "nat \<Rightarrow> ('a \<Rightarrow> 'a) \<Rightarrow> 'a \<Rightarrow> 'a" where 
  "iter 0 f x = x" |
  "iter n f x = f (iter (n-1) f x)"
  
lemma extract_sum:
  fixes m n :: nat
  assumes "m>n"
  shows "(\<Sum>i\<in>{n..<m}. (f i) * q) = (\<Sum>i\<in>{n..<m}. (f i)) * q"
  sorry
    
lemma sum_of_powers:
  fixes m n :: nat
  assumes "m>n"
  shows "(\<Sum>i\<in>{n..<m}. q^i) = q^n*(\<Sum>i\<in>{0..<m-n}. q^i)"
  sorry
    
lemma less_than_series:
  fixes n m :: nat
  assumes "0\<le>q" and "q<1"
    and "m>n"
  shows "(\<Sum>i\<in>{0..<m-n}. q^i) \<le> 1/(1-q)"
  sorry
    
lemma existence:
  assumes "q \<ge> 0" and "q < 1"
  and "r > 0"
  shows "\<exists>p. q^p < r"
  sorry
    
lemma iter_closure:
  assumes "\<And>x. x \<in> carrier \<Longrightarrow> f x \<in> carrier" and "x0 \<in> carrier"
  shows "iter n f x0 \<in> carrier"
proof(induction n)
  show "iter 0 f x0 \<in> carrier"
    using assms by auto
next
  fix n
  assume "iter n f x0 \<in> carrier"
  thus "iter (Suc n) f x0 \<in> carrier"
    using assms by auto
qed
  
lemma iter_collapse:
  assumes "\<And>x. x \<in> carrier \<Longrightarrow> f x \<in> carrier"
  assumes "q\<ge>0" "q < 1" "\<forall>x\<in>carrier. \<forall>y\<in>carrier. metric (f x) (f y) \<le> q * metric x y"
  assumes "x0 \<in> carrier"
  shows "metric (iter (n+1) f x0) (iter n f x0) \<le> q^n * metric (f x0) x0"
    (is "metric (?x (n+1)) (?x n) \<le> q^n * metric (f x0) x0")
proof(induction n)
  show "metric (?x (0 + 1)) (?x 0) \<le> q ^ 0 * metric (f x0) x0"
    by auto
next
    fix n
    assume "metric (?x (n + 1)) (?x n) \<le> q ^ n * metric (f x0) x0"
    hence IH:"q * (metric (?x (n + 1)) (?x n)) \<le> q * (q ^ n * metric (f x0) x0)"
      using assms by (simp add: mult_left_mono)
    have "iter (n + 1) f x0 \<in> carrier" and "iter n f x0 \<in> carrier"
      using assms by (auto simp add: iter_closure)
    hence "metric (f (?x(n + 1))) (f (?x n)) \<le> q * metric (?x(n + 1)) (?x n)"
      using assms by (auto)
    hence "metric (f (?x(n + 1))) (f (?x n)) \<le> q * q ^ n * metric (f x0) x0"
      using IH by auto
    thus "metric (?x (Suc n + 1)) (?x (Suc n)) \<le> q ^ Suc n * metric (f x0) x0"
      by auto
qed
    
lemma iter_inequality:
  assumes "\<And>x. x \<in> carrier \<Longrightarrow> f x \<in> carrier" 
  assumes "q\<ge>0" "q < 1" "\<forall>x\<in>carrier. \<forall>y\<in>carrier. metric (f x) (f y) \<le> q * metric x y"
  assumes "x0 \<in> carrier"
  assumes "m>n"
  shows "metric (iter m f x0) (iter n f x0) \<le> (\<Sum>i\<in>{n..<m}. metric (iter (i+1) f x0) (iter i f x0))"
proof -
      obtain d where "m = d + n"
      proof
        show "m = (m - n) + n"
          using assms by (simp add: less_or_eq_imp_le)
      qed
      have "metric (iter (d+n) f x0) (iter n f x0) \<le> (\<Sum>i\<in>{n..<(d+n)}. metric (iter (i+1) f x0) (iter i f x0))"
      proof(induction d)
        show "metric (iter (0 + n) f x0) (iter n f x0) \<le> (\<Sum>i = n..<0 + n. metric (iter (i + 1) f x0) (iter i f x0))"
        using assms by (metis (no_types, lifting) add.left_neutral add_le_same_cancel2 atLeastLessThan_empty discernible_metric empty_iff iter_closure le_0_eq sum_nonneg)
      next
        fix d
        assume "metric (iter (d + n) f x0) (iter n f x0) \<le> (\<Sum>i = n..<d + n. metric (iter (i + 1) f x0) (iter i f x0))"
        thus "metric (iter (Suc d + n) f x0) (iter n f x0) \<le> (\<Sum>i = n..<Suc d + n. metric (iter (i + 1) f x0) (iter i f x0))"
          using assms by (smt Suc_eq_plus1 add_Suc add_less_same_cancel2 assms(1) diff_add_zero iter_closure le_add1 less_diff_conv less_imp_not_less nat_less_le subadditive_metric sum_op_ivl_Suc)
      qed
      thus "metric (iter m f x0) (iter n f x0) \<le> (\<Sum>i = n..<m. metric (iter (i + 1) f x0) (iter i f x0))"
        using \<open>m = d + n\<close> by blast
qed

lemma iter_cauchy:
  assumes "\<And>x. x \<in> carrier \<Longrightarrow> f x \<in> carrier" 
  assumes "q\<ge>0" "q < 1" "\<forall>x\<in>carrier. \<forall>y\<in>carrier. metric (f x) (f y) \<le> q * metric x y"
  assumes "x0 \<in> carrier"
  assumes "f x0 \<noteq> x0"
  shows "cauchy (\<lambda>n. iter n f x0)"
proof(unfold cauchy_def, safe)
  
  fix \<epsilon> :: real
  assume 1:"\<epsilon> > 0"
  hence a:"\<epsilon> * (1 - q) / (metric (f x0) x0) > 0"
    using assms by (metis diff_gt_0_iff_gt discernible_metric divide_pos_pos less_eq_real_def mult_pos_pos non_negative_metric)
  then obtain p::nat where p:"q^p < \<epsilon> * (1 - q) / (metric (f x0) x0)"
    using assms existence by blast
  {
  fix m n :: nat
  assume "m>p" "n>p" and 1:"m>n"
    
  obtain d where "m = d + n"
  proof
   show "m = (m - n) + n"
     by (simp add: "1" less_imp_le)
   qed
   have "metric (iter m f x0) (iter n f x0) \<le> (\<Sum>i = n..<m. metric (iter (i + 1) f x0) (iter i f x0))"
     using assms 1 iter_inequality by force
   have "metric (iter (d+n) f x0) (iter n f x0) \<le> (\<Sum>i\<in>{n..<(d+n)}. q^i * metric (f x0) x0)"
   proof(induction d)
     show "metric (iter (0 + n) f x0) (iter n f x0) \<le> (\<Sum>i = n..<0 + n. q ^ i * metric (f x0) x0)"
       using assms by (metis (no_types, lifting) add.left_neutral atLeastLessThan_empty 
                              iter_closure discernible_metric empty_iff le_eq_less_or_eq sum_nonneg)
   next
     fix d
     assume "metric (iter (d + n) f x0) (iter n f x0) \<le> (\<Sum>i = n..<d + n. q ^ i * metric (f x0) x0)"
     hence "metric (iter (Suc d + n) f x0) (iter (d + n) f x0) + metric ((iter (d + n) f x0)) (iter n f x0) 
            \<le> (\<Sum>i = n..<d + n. q ^ i * metric (f x0) x0) + metric (iter (Suc d + n) f x0) (iter (d + n) f x0)"
       by linarith
     hence 1:"metric (iter (Suc d + n) f x0)  (iter n f x0) 
            \<le> (\<Sum>i = n..<d + n. q ^ i * metric (f x0) x0) + metric (iter (Suc d + n) f x0) (iter (d + n) f x0)"
       using assms by (smt iter_closure subadditive_metric)
     have "metric (iter (Suc d + n) f x0) (iter (d + n) f x0)  \<le> q^(d+n) * metric (f x0) x0"
       using assms iter_collapse by force
     hence "metric (iter (Suc d + n) f x0)  (iter n f x0) \<le>
            (\<Sum>i = n..<d + n. q ^ i * metric (f x0) x0) + q^(d+n) * metric (f x0) x0"
       using "1" by linarith
     thus "metric (iter (Suc d + n) f x0)  (iter n f x0) \<le>  (\<Sum>i = n..<Suc d + n. q ^ i * metric (f x0) x0)"
       by auto
   qed
   hence "metric (iter m f x0) (iter n f x0) \<le> (\<Sum>i\<in>{n..<m}. (q^i) * metric (f x0) x0)"
     using \<open>m = d + n\<close> by blast
   hence 2:"metric (iter m f x0) (iter n f x0) \<le> (\<Sum>i\<in>{n..<m}. q^i) * metric (f x0) x0"
     using assms 1 by (metis extract_sum)
   have 3:"... = q^n * metric (f x0) x0 * (\<Sum>i\<in>{0..<(m-n)}. q^i)"
     using assms sum_of_powers 1 by fastforce 
   have 4:"... \<le> q^n * metric (f x0) x0 * (1/(1-q))"
     using assms 1 by (meson less_than_series mult_left_mono mult_nonneg_nonneg non_negative_metric zero_le_power)
       
   have 5:"metric (iter m f x0) (iter n f x0) \<le> q^n * metric (f x0) x0 * (1/(1-q))"
     using "2" "3" "4" by linarith
   have "q^n \<le> q^p"
     using assms by (simp add: \<open>p < n\<close>  less_imp_le power_decreasing)
   hence 6:"metric (iter m f x0) (iter n f x0) \<le> q^p * metric (f x0) x0 * (1/(1-q))"
     using 5 assms by (smt divide_pos_pos mult_mono non_negative_metric real_mult_le_cancel_iff1 zero_le_power)
   have "metric (f x0) x0 * (1/(1-q)) > 0"
     using assms by (smt a mult_pos_pos non_negative_metric zero_less_divide_iff)
   hence 7:"metric (iter m f x0) (iter n f x0) < (\<epsilon> * (1 - q) / (metric (f x0) x0)) *  metric (f x0) x0 * (1/(1-q))"
     using "6" mult_strict_right_mono p by fastforce
   hence "... = \<epsilon>"
     using \<open>0 < \<epsilon> * (1 - q) / metric (f x0) x0\<close> by auto
   hence "metric (iter m f x0) (iter n f x0) < \<epsilon>"
     using "7" by auto
  }
  thus" \<exists>p. \<forall>m\<ge>p. \<forall>n\<ge>p . m>n \<longrightarrow> metric (iter m f x0)  (iter n f x0) < \<epsilon>"
    using less_le_trans by blast
qed
  
  

theorem banach_fixed_point:
  assumes "\<And>x. x \<in> carrier \<Longrightarrow> f x \<in> carrier"
   and "contraction_map f"
  assumes "contraction_map f"
  assumes "\<exists>x0. x0 \<in> carrier"
  shows "\<exists>x\<in>carrier. f x = x"
proof -
  obtain x0 where x0:"x0 \<in> carrier"
    using assms by auto
  show ?thesis
  proof(case_tac "f x0 = x0")
    assume "f x0 = x0"
    thus "\<exists>x\<in>carrier. f x = x"
      using x0 by (rule bexI[where x=x0])
  next
    obtain q::real where q:"q\<ge>0" "q<1" "\<forall>x\<in>carrier. \<forall>y\<in>carrier. metric (f x) (f y) \<le> q * metric x y"
      using assms by (auto simp add: contraction_map_def)
  
    assume "f x0 \<noteq> x0"
    hence "cauchy (\<lambda>n. iter n f x0)"
      using assms(1) iter_cauchy q(1) q(2) q(3) x0 by blast
    hence "convergent (\<lambda>n. iter n f x0)"
      using completness by auto
    
    then obtain x where x:"x\<in>carrier" "is_limit (\<lambda>n. iter n f x0) x"
      by (auto simp add: convergent_def)
    hence "is_limit (f \<circ> (\<lambda>n. iter n f x0)) (f x)"
      by (auto simp add: continuous_limits)
    hence "is_limit (\<lambda>n. iter (n+1) f x0) (f x)"
      by (auto simp add:o_def)
    hence step:"is_limit (\<lambda>n. (\<lambda>n. iter n f x0) (n+1)) (f x)"
      by auto
    have "is_limit (\<lambda>n. iter n f x0) (f x) = is_limit (\<lambda>n. (\<lambda>n. iter n f x0) (n+1)) (f x)"
      by (rule limits_preserved)
    hence "is_limit (\<lambda>n. iter n f x0) (f x)"
      using step by auto
    hence  "x = f x"
      using assms x unique_limit x0 by (metis iter_closure)
    thus "\<exists>x\<in>carrier. f x = x"
     using x by force
  qed
qed

end

text\<open>
\begin{center}
\emph{The end\ldots}
\end{center}\<close>
