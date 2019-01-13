theory pratical
  imports Main
begin

(*use rule, erule, and so on combined with basic elimination and introduction rules.  do not use
  any automation...*)
  
lemma
  assumes "P \<subseteq> Q" and "Q \<subset> R"
  shows "P \<subset> R"
  using assms apply -
  apply(rule psubsetI, erule psubsetE, rule subsetI, drule subsetD, assumption, drule subsetD, assumption+)
  apply(rule notI, erule psubsetE, erule equalityE)
  apply(drule subset_trans) back back back
  apply(assumption)
  apply(erule notE, assumption)
 
lemma subset_trans:
  assumes "P \<subseteq> Q" and "Q \<subseteq> R"
  shows "P \<subseteq> R"
  using assms
    apply -
  apply(rule subsetI)
  apply(erule subsetD)
  apply(erule subsetD)
  apply assumption
  done
   
lemma
  shows "P \<subseteq> Q \<longleftrightarrow> P \<inter> Q = P"
  apply(rule iffI)
  apply(rule 
    
lemma
  assumes "x \<in> { x. P x \<and> Q x }"
    and "\<forall>x. P x \<longrightarrow> R x"
  shows "R x"
  using assms apply -
  apply(drule CollectD, erule conjE, erule allE[where x=x], erule impE, assumption+)
    
lemma
  assumes "P \<inter> Q = {}"
  shows "x \<in> P \<longrightarrow> x \<notin> Q"
  using assms apply -
  apply(rule impI, rule notI)
  apply(erule equalityE, drule subsetD, rule IntI, assumption+)
  apply(erule emptyE)


lemma
  shows "P \<union> Q = {} \<longleftrightarrow> P = {} \<and> Q = {}"
  apply(rule iffI)
   apply(rule conjI)
    apply(rule equalityI, erule equalityE, rule subsetI)
    apply(drule subsetD, rule UnI1, assumption+)
    apply(rule subsetI, erule emptyE)
    apply(rule equalityI, erule equalityE, rule subsetI)
    apply(drule subsetD, rule UnI2, assumption+)
    apply(rule subsetI, erule emptyE)
  apply(rule equalityI, rule subsetI, erule conjE, erule UnE)
    apply(erule equalityE, drule subsetD, assumption+)
   apply(erule equalityE) back
   apply(drule subsetD, assumption+)
  apply(rule subsetI, erule emptyE)
    
   
lemma
  shows "P - UNIV = {}"
  apply(rule equalityI)
   apply(rule subsetI, drule DiffD2, rule UNIV_I, assumption)
  apply(rule subsetI, erule emptyE)
    
lemma
  assumes "S \<inter> T = {}"
  shows "(P - S) - T = (P - T) - S"
  using assms apply -
  apply(rule equalityI)
   apply(rule subsetI)
   apply(rule DiffI)+
     apply(drule DiffD1)+
     apply(assumption)
    apply(rule notI, drule DiffD2, assumption+)
   apply(rule notI, drule DiffD1, drule DiffD2, assumption+)
  apply(rule subsetI)
  apply(rule DiffI)+
    apply(drule DiffD1)+
    apply(assumption)
   apply(rule notI, drule DiffD2, assumption+)
   apply(rule notI, drule DiffD1, drule DiffD2, assumption+)
lemma
  assumes "U \<subseteq> S"
    and "U \<subseteq> T"
  shows "U \<subseteq> S \<union> T"
  using assms apply -
  apply(rule subsetI)
  apply(drule subsetD, assumption, rule UnI1, assumption)
  
    
lemma
  assumes "x \<in> (\<Union>i\<in>I. f i)"
  shows "\<exists>i\<in>I. x \<in> f i"
  using assms apply -
  apply(subst (asm) UNION_eq)
  apply(drule CollectD, assumption)
    
lemma
  shows "(\<Union>i\<in>I. f i \<union> g i) = ((\<Union>i\<in>I. f i) \<union> (\<Union>i\<in>I. g i))"
  apply(rule equalityI)
   apply(rule subsetI)
   apply(subst (asm) UNION_eq)
   apply(drule CollectD, erule bexE, erule UnE)
    apply(rule UnI1, subst UNION_eq, rule CollectI, rule bexI, assumption+)
   apply(rule UnI2, subst UNION_eq, rule CollectI, rule bexI, assumption+)
  apply(rule subsetI, subst UNION_eq, erule UnE)
   apply(subst (asm) UNION_eq, drule CollectD, rule CollectI, erule bexE, rule bexI, rule UnI1, assumption+)
   apply(rule CollectI, subst (asm) UNION_eq, drule CollectD, erule bexE, rule bexI, rule UnI2, assumption+)

    
lemma
  assumes "\<exists>i\<in>I. f i = {}"
  shows "(\<Inter>i\<in>I. f i) = {}"
  using assms apply -
  apply(rule equalityI, rule subsetI, erule bexE, subst (asm) INTER_eq, drule CollectD, erule_tac x=i in ballE)
  apply(erule subst, assumption, erule notE, assumption, rule subsetI, erule emptyE)
    
lemma
  shows "f ` S \<subseteq> T \<longleftrightarrow> (\<forall>x\<in>S. f x \<in> T)"
  sorry
    
lemma
  assumes "\<forall>x. f x = x"
  shows "f ` S = S"
  sorry
    
lemma
  shows "(\<forall>x\<in>S\<union>T. f x) \<longleftrightarrow> ((\<forall>x\<in>S. f x) \<and> (\<forall>x\<in>T. f x))"
  sorry
    
lemma
  assumes "x \<in> T"
    and "x \<notin> S"
    and "\<forall>z\<in>S. z \<noteq> x \<longrightarrow> z \<in> T"
  shows "S \<subset> T"
  sorry
    
end