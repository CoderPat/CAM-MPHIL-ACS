theory Assessed_Exercise_One_Questions
  imports Main "~~/src/HOL/Library/Monad_Syntax"
begin
  
section\<open>Parsers, and some useful combinators\<close>

datatype ('a, 'b) parser
  = Parser "'a list \<Rightarrow> ('a list \<times> 'b) set"
  
definition run :: "('a, 'b) parser \<Rightarrow> 'a list \<Rightarrow> ('a list \<times> 'b) set" where
  "run p xs \<equiv> case p of Parser f \<Rightarrow> f xs"

definition fail :: "('a, 'b) parser" where
  "fail \<equiv> Parser (\<lambda>xs. {})"
  
definition succeed :: "'b \<Rightarrow> ('a, 'b) parser" where
  "succeed x \<equiv> Parser (\<lambda>xs. {(xs, x)})"
  
definition choice :: "('a, 'b) parser \<Rightarrow> ('a, 'b) parser \<Rightarrow> ('a, 'b) parser" (infixr "\<oplus>" 65) where
  "choice p1 p2 \<equiv> Parser (\<lambda>xs. run p1 xs \<union> run p2 xs)"
  
subsection\<open>Slightly more complex combinators\<close>
  
definition satisfy :: "('a \<Rightarrow> bool) \<Rightarrow> ('a, 'a) parser" where
  "satisfy p \<equiv>
     Parser (\<lambda>xs.
       case xs of
         [] \<Rightarrow> {}
       | x#xs \<Rightarrow> if p x then {(xs, x)} else {})"
  
definition exact :: "'a \<Rightarrow> ('a, 'a) parser" where
  "exact x \<equiv> satisfy (\<lambda>y. y = x)"
 
(* 6 marks *)  
definition bind :: "('a, 'b) parser \<Rightarrow> ('b \<Rightarrow> ('a, 'c) parser) \<Rightarrow> ('a, 'c) parser" where
  "bind p f \<equiv> Parser (\<lambda>xs. \<Union> ((\<lambda>(q,r). run (f r) q) ` (run p xs) ))"
  
(* 3 marks *)
fun biter :: "nat \<Rightarrow> ('a, 'b) parser \<Rightarrow> ('a, 'b list) parser" where
  "biter 0 p = succeed []" |
  "biter (Suc m) p = bind p (\<lambda>b. bind (biter m p) (\<lambda>bs. succeed (b#bs)))"
  
(* 3 marks *)  
fun exacts :: "'a list \<Rightarrow> ('a, 'a list) parser" where
  "exacts [] = succeed []" |
  "exacts (x#xs) = bind (exact x) (\<lambda>b. bind (exacts xs) (\<lambda>bs. succeed (b#bs)))"  
  
definition map :: "('a, 'b) parser \<Rightarrow> ('b \<Rightarrow> 'c) \<Rightarrow> ('a, 'c) parser" where
  "map p f \<equiv> Parser (\<lambda>xs. (\<lambda>(ys, b). (ys, f b)) ` (run p xs))"
  
(*extra*)
definition flatten :: "('a, ('a, 'b) parser) parser \<Rightarrow> ('a, 'b) parser" where
  "flatten pp \<equiv> Parser(\<lambda>xs. \<Union>( (\<lambda>(ys, p). run p ys) ` (run pp xs)))"

section\<open>Some properties of this library\<close>
  
subsection\<open>Equivalence of parsers\<close>
  
(* 2 marks *)  
definition peq :: "('a, 'b) parser \<Rightarrow> ('a, 'b) parser \<Rightarrow> bool" where
    "peq p q = (\<forall>xs. run p xs = run q xs)"

(* 1 marks *)  
lemma peq_reflexive:
  shows "peq p p"
  apply(simp add: peq_def)
done

  (* 1 marks *)  
lemma peq_symmetric:
  shows "peq p q = peq q p"
  apply(simp add: peq_def)
  apply(auto)
done    

(* 1 marks *)  
lemma peq_transitive:
  assumes "peq p q" and
    "peq q k"
  shows "peq p k"
  using assms apply -
  apply(simp add: peq_def)
  done

(* extra *) 
lemma peq_bind_subs:
  assumes "peq p q"
  shows "peq (bind p f) (bind q f)"
  using assms apply -
  apply(simp add: peq_def run_def bind_def)
done
 
(* extra *)   
lemma peq_choice_subs:
  assumes "peq p q"
  shows "peq (p \<oplus> r) (q \<oplus> r) \<and> peq (r \<oplus> p) (r \<oplus> q)"
  using assms apply -
  apply(simp add: peq_def run_def choice_def)
done

(* extra *) 
lemma peq_eq:
  assumes "peq p q"
  shows "p = q"
  using assms apply -
  apply(simp add:peq_def run_def)
  apply(case_tac p, case_tac q)
  apply(auto)
done
  
    
subsection\<open>Parsers have a commutative monoidal structure under choice\<close>
  
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
    
(* 1 marks *)  
lemma map_id:
  shows "peq (map p (\<lambda>x. x)) p"
  apply(simp add: peq_def)
  apply(simp add: run_def map_def)
done
   
(* 3 marks *)    
lemma map_cov_composition:
  shows "peq (map p (f \<circ> g)) (map (map p g) f)"
  apply(simp add: peq_def)
  apply(simp add: run_def map_def)
  apply(force)
  done
       
(* 2 marks *)    
lemma map_alternative_def:
  shows "peq (map p f) (bind p (\<lambda>x. succeed (f x)))"
  apply(simp add: peq_def)
  apply(simp add: run_def map_def bind_def succeed_def)
  apply(auto)
done

(* extra *)
lemma flatten_alternative_def:
  shows "peq (flatten pp) (bind pp (\<lambda>p. p))"
  apply(simp add:peq_def)
  apply(simp add:run_def flatten_def bind_def)
done
    
(* extra *)
lemma bind_alternative_def:
  shows "peq (bind p f) (flatten (map p f))"
  apply(simp add:peq_def)
  apply(simp add:run_def flatten_def bind_def map_def)
  apply(force)
done
    
subsection\<open>Bind and succeed satisfy the monad laws (and other properties)\<close>
  
(* 1 marks *)   
lemma bind_succeed_neutral:
  shows "peq (bind p (\<lambda>ys. succeed ys)) p"
  apply(simp add: peq_def)
  apply(simp add: bind_def run_def succeed_def)
done  

(* 1 marks *)
lemma bind_succeed_collapse:
  shows "peq (bind (succeed x) f) (f x)"
  apply(simp add: peq_def)
  apply(simp add: run_def bind_def succeed_def)
done

(* 1 marks *)    
lemma bind_assoc:
  shows "peq (bind (bind p f) q) (bind p (\<lambda>x. bind (f x) q))"
  apply(simp add: peq_def)
  apply(simp add: run_def bind_def split_def)
  done
  
(* 1 marks *)
lemma bind_fail_annihil:
  shows "peq (bind fail f) fail"
  apply(simp add: peq_def)
  apply(simp add: run_def bind_def fail_def)
done
    
(* extra *)  
lemma bind_right_fail:
  shows "peq (bind f (\<lambda>_. fail)) fail"
  apply(simp add: peq_def)
  apply(simp add: run_def bind_def fail_def)
done
    
(* 1 marks *)
lemma bind_choice:
  shows "peq (bind (p \<oplus> q) f) (bind p f \<oplus> bind q f)"
  apply(simp add: peq_def)
  apply(simp add: run_def bind_def choice_def)
  done
  
(* 8 marks *)  
lemma bind_interpolate:
  assumes "run (bind p f) xs = ps"
  shows "\<exists>qs. run p xs = qs \<and> ((\<Union>(q, r)\<in>qs. run (f r) q) = ps)"
  using assms apply -
  apply(simp add: run_def bind_def)
done
    
subsection\<open>Properties of iteration\<close>
  
(* helper *) 
lemma bind_assoc_eq:
  shows "(bind (bind p f) q) = (bind p (\<lambda>x. bind (f x) q))"
  apply(rule peq_eq)
  apply(simp add: bind_assoc)
done

(* helper *) 
lemma bind_succeed_collapse_eq:
  shows "(bind (succeed x) f) = (f x)"
  apply(rule peq_eq)
  apply(simp add: bind_succeed_collapse)
done

(* 5 marks *) 
lemma biter_plus_bind:
  shows "peq (biter (m+n) p) (bind (biter m p) (\<lambda>xs. bind (biter n p) (\<lambda>ys. succeed (xs@ys))))"
  apply(induction m, simp add:peq_def run_def bind_def succeed_def)
  apply(simp add:bind_def, fold bind_def)
  apply(subst bind_assoc_eq)+
  apply(subst bind_succeed_collapse_eq)
  apply(simp add:peq_def run_def bind_def succeed_def)
  apply(auto)
done
    
(* 5 marks *)  
lemma exacts_plus_bind:
  shows "peq (exacts (xs@ys)) (bind (exacts (xs)) (\<lambda>r1. bind (exacts (ys)) (\<lambda>r2. succeed(r1@r2))))"
proof(induction xs)
  show "peq (exacts ([] @ ys)) (bind (exacts []) (\<lambda>r1. bind (exacts ys) (\<lambda>r2. succeed (r1 @ r2))))"
    by (simp add: exact_def peq_def run_def bind_def succeed_def)
next
  fix a xs
  assume "peq (exacts (xs@ys)) (bind (exacts (xs)) (\<lambda>r1. bind (exacts (ys)) (\<lambda>r2. succeed(r1@r2))))"
  hence 1:"peq (bind (exact a) (\<lambda>r. bind (exacts (xs @ ys)) (\<lambda>ra. succeed (r # ra))))
               (bind (exact a) (\<lambda>x. bind (exacts xs) (\<lambda>xa. bind (exacts ys) (\<lambda>r. succeed ((x # xa) @ r)))))"
    by (auto simp add: exact_def peq_def run_def bind_def succeed_def)
  hence 2:"peq (exacts (a#xs @ ys))
               (bind (exact a) (\<lambda>x. bind (exacts xs) (\<lambda>xa. bind (exacts ys) (\<lambda>r. succeed ((x # xa) @ r)))))"
    by (simp)
  hence 3:"peq (exacts (a#xs @ ys))
               (bind (exact a) (\<lambda>x. bind (exacts xs) (\<lambda>xa. bind (succeed (x # xa)) (\<lambda>r. bind (exacts ys) (\<lambda>ra. succeed (r @ ra))))))"
    by(simp add: run_def succeed_def bind_def)
  hence 4:"peq (exacts (a#xs @ ys))
               (bind (bind (exact a) (\<lambda>r. bind (exacts xs) (\<lambda>ra. succeed (r # ra)))) (\<lambda>r. bind (exacts ys) (\<lambda>ra. succeed (r @ ra))))"
    by(simp add: run_def bind_def split_def)
  thus "peq (exacts ((a # xs) @ ys)) 
            (bind (exacts (a # xs)) (\<lambda>r1. bind (exacts ys) (\<lambda>r2. succeed (r1 @ r2))))"
    by(simp)
qed 
    
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
binds.def bind_def succeed_def)
  apply(simp add:bind_def, fold bind_def)

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
  
text\<open>Lastly, a sendef bind_def succeed_def)
  apply(simp add:bind_def, fold bind_def)tence in our English-language fragment is a noun phrase followed by a verb phrase.\<close>
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
