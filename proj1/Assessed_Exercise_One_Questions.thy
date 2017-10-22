theory Assessed_Exercise_One_Questions
  imports Main "~~/src/HOL/Library/Monad_Syntax"
begin

section\<open>Introduction\<close>
  
text\<open>This is the first assessed exercise for the L21 ``Interactive Formal Verification'' MPhil
course in academic year 2017--2018.  In this exercise you will write some parser combinators and
thereafter prove some simple properties about these combinators.  The exercise will test your
ability to write recursive and non-recursive functions in Isabelle/HOL, work with sets and
quantifiers, and prove properties about functions using induction and simple automation.
Note that for this first exercise, structured proofs written in Isar are not required, though
for extra credit more ambitious students could try to convert some of their apply-style proofs into
Isar.  Concretely, the marking scheme for this exercise is as follows (out of a total of 100):

  \<^item> \textbf{50 marks} for correct definitions and lemmas, in accordance with the distribution of marks
    outlined in the body of this document,
  \<^item> \textbf{30 marks} for beautiful proofs and definitions, some use of Isar, extra proofs of
    properties about the combinators presented in this exercise, or more useful combinators and
    properties about them, or the use of features of Isabelle not lectured in class.  Any reasonable
    evidence that you have gone ``above and beyond'' in your use of Isabelle will be
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

\paragraph{Before beginning:} for those who have not encountered parser combinators before, it may be
a good idea for you to skim-read Hutton's article on the subject before proceeding with the exercise
to gain some background.\footnote{See \url{http://eprints.nottingham.ac.uk/221/1/parsing.pdf}}
However, the exercises have been written in such a way that consulting any such background material
on combinator parsing is not strictly necessary, with enough background information embedded within
this document to help you through.
\pagebreak\<close>
  
section\<open>Parsers, and some useful combinators\<close>
  
text\<open>Parsing is a common task when developing computer systems.  But what actually are parsers?
Abstractly, parsers take a string of characters and either fail, or return a result derived
from a prefix of that string paired with a continuation string to parse further.  Note here that
``characters'' need not be concrete ASCII or Unicode characters, but may be (and often are) lexical
tokens of the language being parsed, derived from a separate lexing step.
Running with this idea, we can model parsers as functions that take as input an arbitrary list of
generalised characters and return a set of parsed values, paired with continuation lists of
characters yet to parse.  Using this scheme, not only are we parametric in the underlying
character and return value types, but failure and parse ambiguity can be handled uniformly: a
parser returning an empty set of results signals failure, and ambiguity can be handled by simply
returning the set of all possible parse results.

Concretely, in Isabelle/HOL, we can capture this view of parsing as a datatype, like follows:
\<close>
 
datatype ('a, 'b) parser
  = Parser "'a list \<Rightarrow> ('a list \<times> 'b) set"
  
text\<open>Here, the type variable \texttt{'a} stands for the type of elements in the list being
parsed---in most typical applications, usually lexical tokens, ASCII or Unicode characters, as
mentioned---and the type variable \texttt{'b} stands for the return type of the parser.  For
instance, a combinator that parses lists of characters and returns an integer would have type
\texttt{(char, int) parser}, whereas a parser that parser lists of tokens to produce a boolean value
would have type \texttt{(token, bool) parser} for some token type, \texttt{token}.

A parser can be ``run'', or ``executed'' on a list of characters by simply unwrapping it, and
applying the underlying parsing function to produce a set of results:
\<close>
  
definition run :: "('a, 'b) parser \<Rightarrow> 'a list \<Rightarrow> ('a list \<times> 'b) set" where
  "run p xs \<equiv> case p of Parser f \<Rightarrow> f xs"
  
text\<open>There exist two very simple but very important parsers: the parser that always succeeds, and
the parser that always fails.  The parser that always fails is the simpler of the two, so we
consider that parser first.  No matter what its input, this parser always returns an empty set of
results.  This can be modelled easily in Isabelle/HOL, as follows:\<close>

definition fail :: "('a, 'b) parser" where
  "fail \<equiv> Parser (\<lambda>xs. {})"

text\<open>The parser that always succeeds is a little more complex.  For every list of characters to
parse, this parser will simply return a singleton set containing that list as its continuation.
The parser cannot invent a value to return---as we are parametric in the type of values, so have no
idea how to obtain a defined element of type \texttt{'b}---so we must pass it one explicitly,
instead:\<close>
  
definition succeed :: "'b \<Rightarrow> ('a, 'b) parser" where
  "succeed x \<equiv> Parser (\<lambda>xs. {(xs, x)})"
  
text\<open>Lastly, we consider how to make choices when parsing, which is useful: for instance, in a
programming language a number may either be deemed to be a finite list of digits \emph{or} a finite
list of digits preceded by a negation sign.  This sort of choice can easily be captured as a
higher-order function (or combinator), which we call \texttt{choice}.  The combinator takes two
parsers as arguments and produces another parser as a result, and produces its resulting parser by
simply running both argument parsers on the parser's input and collecting together all results using
the set union operator:\<close>
  
definition choice :: "('a, 'b) parser \<Rightarrow> ('a, 'b) parser \<Rightarrow> ('a, 'b) parser" (infixr "\<oplus>" 65) where
  "choice p1 p2 \<equiv> Parser (\<lambda>xs. run p1 xs \<union> run p2 xs)"
  
subsection\<open>Slightly more complex combinators\<close>
  
text\<open>Aside from failure, success, and choice, there are some other commonly reoccurring patterns
that you may notice are common to many parsing tasks, and can therefore be factored out into a
series of reusable combinators.

In particular, when parsing we often want to indicate that the string being parsed must have an
exact form.  This is especially true when parsing keywords or other syntactic elements in a
programming language, for instance.  The following combinator, \texttt{satisfy}, takes a predicate
on characters and produces a parser.  This resulting parser fails if its input is empty, or
else if the supplied predicate does not hold on the first character of its input list. Otherwise it
succeeds, returning a singleton set containing the head of the input list as its value, and the tail
of the input list as its continuation:\<close>
  
definition satisfy :: "('a \<Rightarrow> bool) \<Rightarrow> ('a, 'a) parser" where
  "satisfy p \<equiv>
     Parser (\<lambda>xs.
       case xs of
         [] \<Rightarrow> {}
       | x#xs \<Rightarrow> if p x then {(xs, x)} else {})"
  
text\<open>Admittedly, this combinator is not very interesting on its own.  Rather, it is a combinator
that can be used to derive more interesting combinators.  For instance, parsing an exact character
is now straightforward using the \texttt{satisfy} combinator:\<close>
  
definition exact :: "'a \<Rightarrow> ('a, 'a) parser" where
  "exact x \<equiv> satisfy (\<lambda>y. y = x)"
 
text\<open>Below, we will also see another combinator, \texttt{exacts}, which allows us to parse an entire
string of characters exactly, rather than a single one, which also uses \texttt{exact} as a
subprocedure.
  
Previously, we discussed a parser combinator that captured a notion of ``choice''.  Another
important concept when parsing is sequencing: often we wish to parse a keyword immediately
followed by an identifier, or something similar.  This is a simple notion of sequencing, but
sometimes the pattern of sequencing can be more complex.  For example, we may wish to parse a number
and then parse different things depending on whatever number we obtained from that initial parse.
This is especially true when parsing binary file formats, for instance, which often tell you in
advance in a header entry at the start of the file how many table elements of some sort are to
follow.  This more complex notion of sequencing can be captured by a parser combinator, called
\texttt{bind}.  This combinator takes a parser \texttt{p} as an input and a function \texttt{f}, and
returns a new parser as a result.  Concretely, \texttt{f} accepts the return value of \texttt{p} and
produces a new parser as a result, capturing this idea that sequencing can branch depending on the
intermediate results of previous parses.

\textbf{Exercise (6 marks)}: complete the definition of \texttt{bind} by replacing the
\texttt{consts} declaration below with a complete definition.  Use the type signature of the
function to guide you in its implementation.  Make sure you properly thread all of the continuation
lists of intermediate parses through the definition.  Some of the properties that you will later
prove below may fail to hold should you get the implementation wrong initially, so you may need to
come back and re-examine your implementation later.  You can also use the examples in
Section~\ref{sect.example.parse} to test whether your definition is reasonable.\<close>
  
definition bind :: "('a, 'b) parser \<Rightarrow> ('b \<Rightarrow> ('a, 'c) parser) \<Rightarrow> ('a, 'c) parser" where
  "bind p f \<equiv> Parser (\<lambda>xs. \<Union> ( (\<lambda>(q,r). run (f r) q) ` (run p xs) ))"
  
  
text\<open>Now that we have a notion of sequencing together two parsers to produce a new parser, we can
define a combinator that iteratively applies a parser a fixed number of times in sequence---call it
\texttt{biter}.  This combinator accepts two arguments---call them \texttt{m} and \texttt{p}---where
\texttt{m} is a natural number indicating the number of times to apply parser \texttt{p} in
sequence.  Ideally, we would also like \texttt{biter} to return a list of the elements that it has
parsed, too.

Concretely, what should the \texttt{biter} combinator do?  If we are iterating zero times, then the
resulting parser should succeed, albeit returning the empty list as its value.  Otherwise, if
iterating \texttt{Succ m} times, then we should first parse using $\texttt{p}$ and then immediately
after parse using the \texttt{m}-fold iteration of \texttt{p}, combining the two intermediate
results to produce a list as the final value.

\textbf{Exercise (3 marks)}: complete the definition of the \texttt{biter} combinator by replacing
the \texttt{consts} declaration below with a complete definition.  Again, use the type signature of
the function to guide you in its definition.  Some of the properties that you will later prove below
may fail to hold should you get the implementation wrong initially, so you may need to come back and
re-examine your implementation later.\<close>
  
fun biter :: "nat \<Rightarrow> ('a, 'b) parser \<Rightarrow> ('a, 'b list) parser" where
  "biter 0 p = succeed []" |
  "biter (Suc m) p = bind p (\<lambda>b. Parser (\<lambda>xs. (\<lambda>(ys, bs). (ys, b#bs)) ` (run (biter m p) xs)))"
  
text<   >


text\<open>A slightly different, but related notion of iteration, is often useful.  Suppose we want to
parse a keyword in a programming language.  How can we do that, given the combinators that we
already have?  Using the previously discussed \texttt{exact} combinator, we can parse a single
character of the keyword at a time, in sequence, until there are no more characters of the keyword
left to parse.  The combinator \texttt{exacts} accepts a list of characters \texttt{cs} to parse one
by one, and as its return value it produces the list of characters that it has parsed.

\textbf{Exercise (3 marks)}: complete the definition of the \texttt{exacts} combinator by replacing
the \texttt{consts} declaration below with a complete definition.  Again, use the type signature of
the function to guide you in its definition.  Some of the properties that you will later prove below
may fail to hold should you get the implementation wrong initially, so you may need to come back and
re-examine your implementation later.\<close>
  
fun exacts :: "'a list \<Rightarrow> ('a, 'a list) parser" where
  "exacts [] = succeed []" |
  "exacts (x#xs) = bind (exact x) (\<lambda>a. Parser (\<lambda>ys. (\<lambda>(zs, as). (zs, a#as)) ` (run (exacts xs) ys)))"
  
text\<open>Lastly, suppose we wish to parse numbers for a programming language interpreter we are writing.
One way to do this would be to parse a list of digits, returning a string, and then take this string
and apply a function that maps strings of digits into some numeric type, returning that as our
result.  We can capture this pattern using a notion of ``mapping'', similar to the map function on
lists from standard functional programming:\<close>
  
definition map :: "('a, 'b) parser \<Rightarrow> ('b \<Rightarrow> 'c) \<Rightarrow> ('a, 'c) parser" where
  "map p f \<equiv> Parser (\<lambda>xs. (\<lambda>x. (fst x, f (snd x))) ` (run p xs))"

section\<open>Some properties of this library\<close>
  
text\<open>We now prove some important properties of our parser combinators to ensure that they behave
correctly.  Many of these properties can be expressed as equivalences between parsers.  But first,
we must define what it means for two parsers to be equivalent, or equal.\<close>
  
subsection\<open>Equivalence of parsers\<close>
  
text\<open>What is a suitable notion of equivalence for parsers?  Recall that parsers are essentially
functions in disguise, and as you know functions are equal when they agree on all inputs.  Two
parsers should therefore be considered equivalent when executing them both on the same arbitrary
input leads to the same result.

\textbf{Exercise (2 marks)}: define a binary relation \texttt{peq} on parsers that captures when two
parsers are equivalent by replacing the \texttt{consts} declaration below with a complete definition.
Some of the properties that you will later prove below may fail to hold should you get the
implementation wrong initially, so you may need to come back and re-examine your implementation
later.\<close>
  
definition peq :: "('a, 'b) parser \<Rightarrow> ('a, 'b) parser \<Rightarrow> bool" where
    "peq p q = (\<forall>xs. run p xs = run q xs)"

text\<open>We can check that the putative equivalence relation above behaves somewhat correctly by
checking that it is indeed an equivalence relation, i.e. that it is reflexive, symmetric, and
transitive.

\textbf{Exercise (3 marks, 1 mark each)}: prove that \texttt{peq} is \emph{reflexive},
\emph{symmetric}, and \emph{transitive} by stating and proving three relevant lemmas.\<close>
  
lemma peq_reflexive:
  shows "peq p p"
  apply(simp add: peq_def)
done

lemma peq_symmetric:
  shows "peq p q = peq q p"
  apply(simp add: peq_def)
  apply(auto)
done    
    
lemma peq_transitive:
  assumes "peq p q" and
    "peq q k"
  shows "peq p k"
  using assms apply -
  apply(simp add: peq_def)
  done
    
lemma peq_bind_substitution:
  assumes "peq p q"
  shows "peq (bind p f) (bind q f)"
  using assms apply -
  apply(simp add: peq_def run_def bind_def)
done
    
subsection\<open>Parsers have a commutative monoidal structure under choice\<close>
  
text\<open>Now we have a notion of parser equivalence, we can state and prove some interesting properties
of our combinators.  First, we examine how \texttt{choice} and the always-failing parser,
\texttt{fail}, interact.  It should be intuitively obvious that \texttt{fail} acts as a
\emph{neutral} (or identity) element for the \texttt{choice} combinator.

\textbf{Exercise (2 marks, 1 mark each)}: prove that \texttt{fail} is a left- and right-neutral
element for \texttt{choice} by proving the following two lemmas.  That is, replace the \texttt{oops}
commands below with complete proofs.\<close>
  
(* 1 marks *)
lemma choice_ident_fail1:
  shows "peq (fail \<oplus> p1) p1"
  apply(simp add: peq_def)
  apply(simp add: choice_def run_def fail_def)
done
    
(* 1 marks *)
lemma choice_ident_fail2:
  shows "peq (p1 \<oplus> fail) p1"
  apply(simp add: peq_def)
  apply(simp add: run_def choice_def fail_def)
done
    
text\<open>Moreover, it should be obvious that choice is commutative and associative---it does not and
should not matter in which order you choose to apply parsers under the choice combinator, as the
choice combinator simply collects together all possible parses in one big set.

\textbf{Exercise (2 marks, 1 mark each)}: prove that \texttt{choice} is commutative and associative
by proving the following two lemmas.  That is, replace the \texttt{oops} commands below with
complete proofs.\<close>

(* 1 marks *)
lemma choice_comm:
  shows "peq (p1 \<oplus> p2) (p2 \<oplus> p1)"
  apply(simp add: peq_def)
  apply(simp add: run_def choice_def)
  apply(auto)
done
    
(* 1 marks *)
lemma choice_assoc:
  shows "peq (p1 \<oplus> (p2 \<oplus> p3)) ((p1 \<oplus> p2) \<oplus> p3)"
  apply(simp add: peq_def)
  apply(simp add: run_def choice_def)
  apply(auto)
done
    
subsection\<open>Map is functorial, and has an alternative definition\<close>
  
text\<open>The familiar ``map'' function on lists enjoys a number of properties.  A few of these
properties are particularly important, namely:

  \<^item> mapping the identity function $id$ over a list results in the same list,
  \<^item> mapping the composition of two functions, $f \circ g$, over a list is the same as first mapping
    $g$ over that list followed by mapping $f$ over the resulting list.

These two properties taken together are sometimes known as ``functoriality'', and a great number of
similar ``map'' functions on different types possess them.  Indeed, the map function that we have
defined on parsers also possesses these functoriality properties.

\textbf{Exercise (1 mark)}: show that mapping the identity function over a parser \texttt{p} is
equivalent to \texttt{p} by stating and proving a relevant lemma.

\textbf{Exercise (3 marks)}: show that mapping the composition of two functions over a parser is
equivalent to first mapping one function, and then the other, over that same parser by stating and
proving a relevant lemma.\<close>
  
text\<open>Earlier we gave a direct definition of \texttt{map} in terms of the set image function.  It was
rather ``low level'', requiring us to deal directly with the underlying representation of parsers.
In fact, having defined the \texttt{bind} and \texttt{succeed} combinators already, it was already
possible to give an alternative definition of \texttt{map} in terms of those operations that did not
require us to deal directly with the underlying representation of parsers.

\textbf{Exercise (2 marks)}: show that \texttt{map} can be given an alternative definition in terms
of the \texttt{bind} and \texttt{succeed} combinators by proving the following lemma.  That is,
replace the \texttt{oops} command below with a complete proof.\<close>
    
(* 2 marks *)    
lemma map_alternative_def:
  shows "peq (map p f) (bind p (\<lambda>x. succeed (f x)))"
  apply(simp add: peq_def)
  apply(simp add: run_def map_def bind_def succeed_def split_def)
  apply(auto)
done
  
    
subsection\<open>Bind and succeed satisfy the monad laws (and other properties)\<close>
    
text\<open>Next, we examine how the \texttt{bind} and \texttt{succeed} combinators interact with each
other.  First, we show that \texttt{succeed} acts as a right-neutral element for \texttt{bind}.

\textbf{Exercise (1 mark)}: show that \texttt{succeed} is a right-neutral element for \texttt{bind}
by stating and proving a relevant lemma.\<close>
  
text\<open>Moreover, succeed also acts as a kind of left-neutral element for bind, albeit in a slightly
messier way than for the right-neutral case.  As a result, I will provide the lemma statement.

\textbf{Exercise (1 mark)}: show that \texttt{succeed} is a left-neutral element for \texttt{bind}
by proving the following lemma.  That is, replace the \texttt{oops} command below with a complete
proof.\<close>

(* 1 marks *)
lemma bind_succeed_collapse:
  shows "peq (bind (succeed x) f) (f x)"
  apply(simp add: peq_def)
  apply(simp add: run_def bind_def succeed_def)
done

text\<open>The \texttt{bind} combinator also exhibits a kind of ``associativity'' property which allows
one to rearrange a series of nested applications of \texttt{bind} from being left-associative to
right-associative.  Again, this is a rather messy property, so I will provide the lemma statement
to prove.

\textbf{Exercise (1 mark)}: show that \texttt{bind} has an associativity property by proving the
following lemma.  That is, replace the \texttt{oops} command below with a complete proof.\<close>

(* 1 marks *)    
lemma bind_assoc [simp]:
  shows "peq (bind (bind p f) q) (bind p (\<lambda>x. bind (f x) q))"
  apply(simp add: peq_def)
  apply(simp add: run_def bind_def split_def)
  done

    
text\<open>(The previous three properties are sometimes referred to as the ``monad laws'', and hold for
many useful parameterised types that appear naturally in functional programming with suitable
definitions for \texttt{bind} and \texttt{succeed}.  For example: lists, sets, the option type,
continuations, and so on, all possess the three properties above for suitably chosen implementations
of \texttt{bind} and \texttt{succeed} specific to each type.)

The \texttt{bind} combinator also satisfies some other properties not captured by the monad laws.
In particular, it interacts well with the always-failing parser, \texttt{fail}, and in fact
\texttt{fail} is a left-annihilating element for \texttt{bind}.  Intuitively, that is: if you first
fail to parse anything, and then try to parse something else, you will always fail.

\textbf{Exercise (1 mark)}: show that \texttt{fail} is a left-annihilating element for \texttt{bind}
by proving the following lemma.  That is, replace the \texttt{oops} command below with a complete
proof.\<close>
  
(* 1 marks *)
lemma bind_fail_annihil:
  shows "peq (bind fail f) fail"
  apply(simp add: peq_def)
  apply(simp add: run_def bind_def fail_def)
done
    
text\<open>In addition, the \texttt{bind} and \texttt{choice} combintors also interact well, and one may
factor \texttt{bind} through the \texttt{choice} combinator freely.

\textbf{Exercise (1 mark)}: show \texttt{bind} can be factored through the \texttt{choice}
combinator by proving the following lemma.  That is, replace the \texttt{oops} command below with a
complete proof.\<close>
    
(* 1 marks *)
lemma bind_choice_split:
  shows "peq (bind (p \<oplus> q) f) (bind p f \<oplus> bind q f)"
  apply(simp add: peq_def)
  apply(simp add: run_def bind_def choice_def)
  done
  
    
text\<open>Lastly, \texttt{bind} has a rather strong interpolation property that not all parameterised
types that satisfy the monad laws possess (though others, such as the familiar option type do
possess a very similar property).

\textbf{Exercise (8 marks)}: show that \texttt{bind} exhibits an interpolation property by proving
the following lemma.  That is, replace the \texttt{oops} command below with a complete proof.\<close>
  
(* 8 marks *)
lemma bind_interpolate [simp]:
  assumes "run (bind p f) xs = ps"
  shows "\<exists>qs. run p xs = qs \<and> ((\<Union>(q, r)\<in>qs. run (f r) q) = ps)"
  using assms apply -
  apply(simp add: run_def bind_def)
done
    
subsection\<open>Properties of iteration\<close>
  
text\<open>In a final pair of exercises, we address properties of the iteration combinators that we
defined previously.

First, we show that iterating a parser $m+n$ times is the same as iterating a parser $m$ times
followed by iterating a parser $n$ times, before succeeding with the append of the two results.

\textbf{Exercise (5 marks)}: show that this property holds of the \texttt{biter} combinator by
proving the following lemma.  That is, replace the \texttt{oops} command below with a complete
proof.\<close>

(* 5 marks *) 
  
lemma bitter_plus_one_inf:
  assumes "peq (bind (biter (Suc 0) p) (\<lambda>xs. bind (biter m p) (\<lambda>ys. succeed (xs@ys)))) q"
  shows "peq (biter (Suc m) p) q"
  using assms apply -
  apply(subst peq_symmetric)
  apply(subst (asm) peq_symmetric)
  apply(simp add: peq_def bind_def run_def succeed_def)
  apply(case_tac m, simp add:succeed_def)
  apply(simp add:split_def bind_def, auto)
  done
    
lemma bitter_plus_one:
  shows "peq (biter (Suc m) p) (bind (biter (Suc 0) p) (\<lambda>xs. bind (biter m p) (\<lambda>ys. succeed (xs@ys))))"
  apply(simp add: peq_def bind_def run_def succeed_def)
  apply(case_tac m, simp add:succeed_def)
  apply(simp add:split_def bind_def, auto)
done

lemma peq_bind_subst_inf:
  assumes "peq p q" and "f = g" and "peq r (bind p f)" 
  shows "peq r (bind q g)"
  using assms apply -
  apply(simp add: peq_def run_def bind_def)
  done

lemma peq_bind_right_subst:
  assumes "f=g"
  shows "peq (bind p f) (bind p g)"
  using assms apply -
  apply(simp add: peq_def run_def bind_def)
  done 
    
lemma bind_assoc_implication:
  assumes "peq r (bind p (\<lambda>x. bind (f x) q))"
  shows "peq r (bind (bind p f) q)"
  using assms apply -
  apply(simp add: peq_def run_def bind_def split_def)
  done
    
lemma biter_plus_bind:
  shows "peq (biter (m+n) p) (bind (biter m p) (\<lambda>xs. bind (biter n p) (\<lambda>ys. succeed (xs@ys))))"
  apply(induction m , simp add:peq_def run_def bind_def succeed_def)
  apply(subst Nat.plus_nat.add_Suc)
  apply(rule bitter_plus_one_inf)
  using bitter_plus_one[where p=p] apply -
  apply(rule_tac p="(bind (biter (Suc 0) p) (\<lambda>xs. bind (biter m p) (\<lambda>ys. succeed (xs@ys))))" in peq_bind_subst_inf)
    apply(simp add:peq_symmetric)
  apply(simp)
  apply(rule bind_assoc_implication, rule peq_bind_right_subst)
    
lemma biter_plus_bind:
  shows "peq (biter (m+n) p) (bind (biter m p) (\<lambda>xs. bind (biter n p) (\<lambda>ys. succeed (xs@ys))))"
  apply(induction m, simp add:peq_def run_def bind_def succeed_def)
  apply(simp add: peq_def run_def bind_def succeed_def split_def)
    
find_theorems "(\<forall>x. (?f x = ?g x))" 

  


  
    
   
text\<open>A very similar property holds of the combinator \texttt{exacts}.  If we try to exactly parse a
keyword \texttt{xs} appended to another keyword \texttt{ys}, then this should be the same as exactly
parsing \texttt{xs} followed by exactly parsing \texttt{ys}, succeeding with the append of the two
intermediate results as the return value.

\textbf{Exercise (5 marks)}: show that this analogous described property holds for the
\texttt{exacts} combinator by stating and proving a relevant lemma.\<close>
    
section\<open>Example: parsing a fragment of English\label{sect.example.parse}\<close>
  
text\<open>This section is non-assessed, and is included to provide a motivating example, so that you can
gain some intuition for what the definitions should do, and as a testing ground for you to use to
ensure that your definitions are correct.  In particular, we will use our small combinator library
to parse a tiny (but ambiguous) fragment of English.

The following command is used to set up the ``monadic do''-syntax, which allows us to write repeated
binds as a ``do block'' in the form \texttt{do \{ \ldots \}}.  This command can be safely ignored,
as it only makes the rest of the material below easier to read.\<close>

adhoc_overloading Monad_Syntax.bind bind

text\<open>First, we define a small utility combinator that exactly parses an arbitrary character from a
supplied set of characters.  This will be used below.\<close>

definition one_of :: "'a set \<Rightarrow> ('a, 'a) parser" where
  "one_of ss \<equiv> satisfy (\<lambda>x. x \<in> ss)"
  
text\<open>Next, we give ourselves a supply of common English nouns, verbs, transitive verbs, and
determinants.  These are the basic building blocks of the sentences that we will try to parse.  Note
here that the English word ``loves'' is classed as both a transitive verb and a plain verb,
indicating a degree of ambiguity when parsing is to be expected.\<close>
  
definition nouns :: "string set" where
  "nouns \<equiv> {''man'', ''woman'', ''child''}"
  
definition verbs :: "string set" where
  "verbs \<equiv> {''runs'', ''walks'', ''loves''}"
  
definition transitive_verbs :: "string set" where
  "transitive_verbs \<equiv> {''likes'', ''loves''}"
  
definition determinants :: "string set" where
  "determinants \<equiv> {''a'', ''the'', ''some'', ''every''}"
  
text\<open>A noun phrase in English is a determinant followed by a noun.  Note that the sequencing between
the two is expressed using \texttt{bind}, albeit hidden beneath the \texttt{do \{ ... \}} syntax.\<close>
definition noun_phrase :: "(string, string list) parser" where
  "noun_phrase \<equiv>
     do
     { d \<leftarrow> one_of determinants
     ; n \<leftarrow> one_of nouns
     ; succeed [d, n]
     }"
  
text\<open>An alternative, much less readable rendering of \texttt{noun\_phrase} above, which does not
use the \texttt{do \{ ... \}} syntax is:\<close>
  
definition noun_phrase' :: "(string, string list) parser" where
  "noun_phrase' \<equiv> bind (one_of determinants) (\<lambda>d. bind (one_of nouns) (\<lambda>n. succeed [d, n]))"
  
value "run noun_phrase [''the'', ''woman'']"
  
text\<open>From this, it should be intuitively clear how a \texttt{do}-block is translated into nested
binds.

A verb phrase in English is either a verb, or a transitive verb followed by a noun phrase.
Note here that this example uses \emph{both} choice and sequencing.\<close>
definition verb_phrase :: "(string, string list) parser" where
  "verb_phrase \<equiv>
     (do
     { v \<leftarrow> one_of verbs
     ; succeed [v]
     }) \<oplus> 
     (do
     { t \<leftarrow> one_of transitive_verbs
     ; n \<leftarrow> noun_phrase
     ; succeed (t#n)
     })"
  
text\<open>Lastly, a sentence in our English-language fragment is a noun phrase followed by a verb phrase.\<close>
definition sentence :: "(string, string list) parser" where
  "sentence \<equiv>
     do { n \<leftarrow> noun_phrase
        ; v \<leftarrow> verb_phrase
        ; succeed (n@v)
        }"
  
text\<open>Note in all cases above, when parsing a fragment of English, our parsers return the words
(or a list of them) that were parsed as their return value.  We can now test our parsers, to make
sure they are behaving as expected.  First, some sentences that should be successfully parsed:\<close>
  
value "run sentence [''some'', ''man'', ''likes'', ''the'', ''woman'']"
value "run sentence [''some'', ''child'', ''walks'']"

text\<open>Ambiguous sentences should also work fine.  Note that all parses should be returned, and the
continuation lists should look ``correct'':\<close>
value "run sentence [''some'', ''woman'', ''loves'', ''a'', ''child'']"

text\<open>Multiple sentences can be parsed using iteration.  Here we parse two consecutive sentences.
Again all possible parses of the sentences should be returned:\<close>
value "run (biter 2 sentence) [''some'', ''woman'', ''loves'', ''a'', ''child'',
  ''every'', ''man'', ''loves'', ''a'', ''child'']"
  
text\<open>Here is a parse that should fail (that is, return an empty list of results):\<close>
value "run sentence [''some'', ''man'', ''hates'', ''a'', ''horse'']"
  
text\<open>Note that none of these examples will properly evaluate to a set of value-continuation pairs
until you supply correct definitions in the exercises above.\<close>
  

end
