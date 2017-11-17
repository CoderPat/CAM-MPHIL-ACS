
section\<open>Introduction\<close>
  
section\<open>Metric spaces, and some examples\<close> 
  
theory Assessed_Exercise_Two_Questions
  imports Complex_Main
begin
  
 
locale metric_space =
  fixes carrier :: "'a set"
    and metric :: "'a \<Rightarrow> 'a \<Rightarrow> real"
  assumes non_negative_metric:"(\<forall>x\<in>carrier . \<forall>y\<in>carrier . metric x y \<ge> 0)"
     and  reflexive_metric:   "(\<forall>x\<in>carrier . \<forall>y\<in>carrier . metric x y = metric y x)"
     and  discernible_metric: "(\<forall>x\<in>carrier . \<forall>y\<in>carrier . metric x y = 0 \<longleftrightarrow> x = y)"
     and  subadditive_metric: "(\<forall>x\<in>carrier . \<forall>y\<in>carrier . \<forall>z\<in>carrier.
                                  metric x z \<le> metric x y + metric y z)"
  

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
  
context fixes domain :: "'a set" and metric_d :: "'a \<Rightarrow> 'a \<Rightarrow> real"
       and  codomain :: "'b set" and metric_c :: "'b \<Rightarrow> 'b \<Rightarrow> real"
begin   
  
definition  continuous_at :: "('a \<Rightarrow> 'b) \<Rightarrow> 'a \<Rightarrow> bool" where
  "continuous_at f a \<equiv> \<forall>\<epsilon>>0. \<exists>d>0. (\<forall>x\<in>domain. metric_d x a < d \<longrightarrow> metric_c  (f x) (f a) < \<epsilon>)"
  
definition  continuous :: "('a \<Rightarrow> 'b) \<Rightarrow> bool" where
  "continuous f \<equiv> \<forall>x\<in>domain. continuous_at f x"

end
 
term continuous_at
term continuous

(* 3 marks *) 
lemma continuous_id:
  assumes "metric_space domain metric_d"
  shows "continuous domain metric_d metric_d (\<lambda>x. x)"
proof(unfold continuous_def continuous_at_def, safe)
  fix x1
  fix \<epsilon> :: real
  assume 1: "0 < \<epsilon>"
  fix x2
  assume "x2 \<in> domain"
  {
    assume "metric_d x2 x1 < \<epsilon>"
    hence "metric_d x2 x1 < \<epsilon>"
      by auto
  }
  thus "\<exists>d>0. \<forall>x2\<in>domain. metric_d x2 x1 < d \<longrightarrow> metric_d x2 x1 < \<epsilon>"
    using 1 by auto
qed

(*4 marks *) 
lemma continuous_const:
  assumes "metric_space domain metric_d" and "metric_space codomain metric_c"
    and "y \<in> codomain"
  shows "continuous domain metric_d metric_c (\<lambda>x. y)"
proof(unfold continuous_def continuous_at_def, safe)
  fix x1
  fix \<epsilon> :: real
  assume 1: "0 < \<epsilon>"
  fix x2
  assume "x2 \<in> domain"
  {
    assume "metric_d x2 x1 < \<epsilon>"
    have "metric_c y y = 0"
      using assms by (auto simp add: metric_space_def)
    hence "metric_c y y < \<epsilon>"
      using 1 by auto
  }
  thus "\<exists>d>0. \<forall>x2\<in>domain. metric_d x2 x1 < d \<longrightarrow> metric_c y y < \<epsilon>"
    using 1 assms metric_space.discernible_metric by force
qed

(* 6 marks *) 
lemma continuous_comp:
  assumes "metric_space domain metric_d" and "metric_space inters metric_i"
    and "metric_space codomain metric_c"
    and "continuous domain metric_d metric_i f"
    and "continuous inters metric_i metric_c g"
    and "\<And>x. x \<in> domain \<Longrightarrow> f x \<in> inters"
  shows "continuous domain metric_d metric_c (g o f)"
proof(simp add:continuous_def continuous_at_def, safe)
  fix x1
  fix \<epsilon> :: real
  assume 1:"\<epsilon> > 0"
  assume 2:"x1 \<in> domain" 
  hence 3:"(f x1) \<in> inters"
    using assms by auto
  hence "\<exists>d>0. \<forall>fx\<in>inters. metric_i fx (f x1) < d \<longrightarrow> metric_c (g fx) (g (f x1)) < \<epsilon>"
    using 1 assms by (auto simp add:continuous_def continuous_at_def)
  then obtain k::real where 4:"k>0" 
              and 5:"\<forall>x2\<in>domain. metric_i (f x2) (f x1) < k \<longrightarrow> metric_c (g (f x2)) (g (f x1)) < \<epsilon>"
    using assms by blast
   
  have "\<exists>d>0. \<forall>x2\<in>domain. metric_d x2 x1 < d \<longrightarrow> metric_i (f x2) (f x1) < k"
    using 2 4 assms by (auto simp add:continuous_def continuous_at_def)
  then obtain d::real where 6:"d>0"
                and 7:"\<forall>x2\<in>domain. metric_d x2 x1 < d \<longrightarrow> metric_i (f x2) (f x1) < k"
    by auto
  fix x2
  assume x2:"x2 \<in> domain"
  {
    assume "metric_d x2 x1 < d"
    hence "metric_c (g (f x2)) (g (f x1)) < \<epsilon>"
      using 5 7 x2 by simp
  }
  thus "\<exists>d>0. \<forall>x2\<in>domain. metric_d x2 x1 < d \<longrightarrow> metric_c (g (f x2)) (g (f x1)) < \<epsilon>"
    using "5" "6" "7" by blast
qed
  
section\<open>Open balls\<close>
  
(* 2 marks *) 
context metric_space begin
  
definition open_ball :: "'a \<Rightarrow> real \<Rightarrow> 'a set" where
  "open_ball c r \<equiv> { x \<in> carrier. metric c x < r }"
    
end
  
context metric_space begin

(* 2 marks *)  
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
  
section\<open>Cauchy, Limits and Convergence\<close>
  
context metric_space
begin

definition cauchy :: "(nat \<Rightarrow>'a) \<Rightarrow> bool"  where
  "cauchy seq \<equiv> (\<forall>\<epsilon>>0. \<exists>p. \<forall>m\<ge>p. \<forall>n\<ge>p . m>n \<longrightarrow> metric (seq m) (seq n) < \<epsilon>)"

definition is_limit :: "(nat \<Rightarrow>'a) \<Rightarrow> 'a \<Rightarrow> bool"where
  "is_limit seq l \<equiv> \<forall>\<epsilon>>0. \<exists>p. \<forall>n\<ge>p. metric (seq n) l < \<epsilon>"
  
lemma unique_limit:
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
 
lemma continuous_limits:
  assumes "\<And>x. x \<in> carrier \<Longrightarrow> f x \<in> carrier"
  assumes "\<forall>n. seq n \<in> carrier" "l \<in> carrier"
  assumes "continuous_at carrier metric metric f l" 
  assumes "is_limit seq l"
  shows "is_limit (f \<circ> seq) (f l)"
proof(unfold is_limit_def, unfold o_def, standard, rule impI)
  fix \<epsilon> :: real
  assume \<epsilon>:"\<epsilon> > 0"
  then obtain d where d:"d>0" "\<forall>x \<in> carrier. metric x l < d \<longrightarrow> metric (f x) (f l) < \<epsilon>"
    by (metis assms(4) \<epsilon> continuous_at_def)
  hence "\<exists>p. \<forall>n\<ge>p. metric (seq n) l < d"
    using assms(5) is_limit_def by blast
  then obtain p where p:"\<forall>n\<ge>p. metric (seq n) l < d"
    by auto
  {
  fix n :: nat
  assume n:"n\<ge>p"
  have "metric (f (seq n)) (f l) < \<epsilon>"
    using assms(2) d(2) n p by auto
  }
  thus "\<exists>p. \<forall>n\<ge>p. metric (f (seq n)) (f l) < \<epsilon>"
    by blast
qed
  
lemma limits_preserved:
  shows "is_limit seq l = is_limit (\<lambda>n. seq (n+1)) l"
proof(unfold is_limit_def, standard, safe)
  assume ass:"\<forall>\<epsilon>>0. \<exists>p. \<forall>n\<ge>p. metric (seq n) l < \<epsilon>"
  fix \<epsilon> :: real
  assume \<epsilon>:"\<epsilon> > 0"
  then obtain p where p:"\<forall>n\<ge>p. metric (seq n) l < \<epsilon>"
    using \<epsilon> is_limit_def ass by blast
  {
    fix n :: nat
    assume "n \<ge> p"
    hence "n+1 \<ge> p"
      by linarith
    hence "metric (seq (n+1)) l < \<epsilon>"
      using p by blast 
  }
  thus "\<exists>p. \<forall>n\<ge>p. metric (seq (n + 1)) l < \<epsilon>"
    by auto
next
  assume ass:"\<forall>\<epsilon>>0. \<exists>p. \<forall>n\<ge>p. metric (seq (n+1)) l < \<epsilon>"
  fix \<epsilon> :: real
  assume \<epsilon>:"\<epsilon> > 0"
  then obtain p where p:"\<forall>n\<ge>p. metric (seq (n+1)) l < \<epsilon>"
    using \<epsilon> is_limit_def ass by blast
  obtain p2 where p2:"p2 = p + 1"
    by auto
  {
    fix n :: nat
    assume n:"n \<ge> p2"
    obtain m where m:"m = n - 1"
      by auto
    hence "m \<ge> p"
      using n p2 by linarith
    hence "metric (seq (m+1)) l < \<epsilon>"
      using p by blast 
    hence "metric (seq n) l < \<epsilon>"
      using m n p2 by (metis add_leD2 le_add_diff_inverse2)
  }
  thus "\<exists>p. \<forall>n\<ge>p. metric (seq n) l < \<epsilon>"
    by auto
qed
  
      
definition convergent where
  "convergent seq \<equiv> (\<exists>x0\<in>carrier. is_limit seq x0)"
   
lemma convergent_cauchy:
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
  "contraction_map f \<equiv> (\<exists>q\<ge>0. q<1 \<and> 
                              (\<forall>x\<in>carrier. \<forall>y\<in>carrier. metric (f x) (f y) \<le> q * metric x y))"
  
lemma contraction_continuous:
  assumes "\<And>x. x \<in> carrier \<Longrightarrow> f x \<in> carrier"
  and "contraction_map f"
shows "continuous carrier metric metric f"
proof(unfold continuous_def continuous_at_def, safe)
  fix x 
  assume x:"x \<in> carrier"
  fix \<epsilon> :: real
  assume \<epsilon>:"\<epsilon>>0"
  then obtain q::real where q:"q\<ge>0" "q<1" "\<forall>y\<in>carrier. metric (f x) (f y) \<le> q * metric x y"
    using x assms contraction_map_def by blast
  obtain d where d:"d = \<epsilon>/q"
    by auto
  {
    fix x2
    assume x2:"x2\<in>carrier" "metric x2 x < d"
    have "metric (f x2) (f x) \<le> q * metric x2 x"
      using assms(1) q(3) reflexive_metric x x2(1) by auto
    hence "metric (f x2) (f x) < q * d"
      using d x x2 q by (metis division_ring_divide_zero less_le_trans 
                           linordered_comm_semiring_strict_class.comm_mult_strict_left_mono 
                           non_negative_metric not_less_iff_gr_or_eq)
    hence "metric (f x2) (f x) < \<epsilon>"
      using \<epsilon> d less_le_trans by fastforce
  }
  thus "\<exists>d>0. \<forall>xa\<in>carrier. metric xa x < d \<longrightarrow> metric (f xa) (f x) < \<epsilon>"
    using d \<epsilon> assms q x by (metis divide_pos_pos dual_order.order_iff_strict 
                                  le_less_trans mult_zero_left reflexive_metric)
qed
  
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
      using assms x0 by (metis continuous_def continuous_limits iter_closure contraction_continuous)
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
