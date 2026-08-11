、# 柑橘（Citrus）组学研究前沿：从参考基因组到系统生物学

## 摘要

柑橘属（Citrus L.）是全球最重要的果树类群之一，其果实、精油与次生代谢产物在农业与食品工业中具有重要价值。过去十余年，高通量测序与多组学技术的飞速发展，推动柑橘研究从单基因水平走向全基因组尺度的系统解析。本文以组学技术演进为主线，系统综述柑橘组学研究前沿：基因组学层面，回顾从2013年甜橙（Citrus sinensis (L.) Osbeck）首个参考基因组草图到端粒到端粒（telomere-to-telomere, T2T）组装、单倍型解析基因组与超级泛基因组的发展历程，并总结群体基因组学对柑橘驯化与网状进化历史的揭示；转录组与表观组学层面，介绍高时空分辨转录组、激光显微切割与染色质动态调控在果实发育与品质形成中的应用；代谢组学与多组学整合层面，概述苯丙烷、类黄酮与类胡萝卜素等次生代谢通路的多组学解析策略；单细胞与空间组学层面，评述单核转录组（snRNA-seq）与质谱成像（MSI）等新技术的应用潜力；系统生物学层面，以黄龙病（Huanglongbing, HLB）为典型案例，整合多组学证据阐释病原-寄主互作、免疫调控与耐受机制。最后，本文讨论了当前柑橘组学研究的不足与未来方向，以期为柑橘基础研究与分子育种提供参考。

## 关键词

柑橘；基因组学；单倍型解析基因组；泛基因组；转录组；表观组学；代谢组学；单细胞转录组；空间组学；黄龙病

## 1 引言

柑橘属（Citrus L.）隶属于芸香科（Rutaceae），是全球最重要的果树类群之一。柑橘果实既可鲜食也可加工，果皮与种子富含类黄酮、类胡萝卜素、苯丙烷类等生物活性物质，兼具营养与保健价值。柑橘类果树广泛栽培于热带、亚热带地区，在全球农业经济中占有举足轻重的地位[1,2]。然而，柑橘属的物种形成与驯化历史高度复杂：栽培种多源于种间杂交与渐渗，基因组杂合度高，多倍化与无融合生殖（珠心胚）现象普遍，这给传统遗传学研究与育种带来了巨大挑战[1,3,4]。与此同时，由韧皮部限制性细菌'Candidatus Liberibacter asiaticus'（CLas）引起的黄龙病（Huanglongbing, HLB）正持续威胁全球柑橘产业，其防控亟需对病原-寄主分子互作的系统认识[5]。

组学技术的进步为破解上述难题提供了前所未有的工具。自2013年甜橙（Citrus sinensis (L.) Osbeck）首个参考基因组草图发布以来[6]，柑橘基因组学经历了从草图到染色体水平、从单一参考基因组到单倍型解析与T2T组装、从核心泛基因组到超级泛基因组的快速演进[1,7,8]。与此同时，高时空分辨转录组[9]、表观组学[10]、单核转录组[11]与纵向多组学[12]等技术的发展，正在将柑橘研究推向系统生物学的新高度。

本文在系统检索相关文献的基础上，围绕五个主题综述柑橘组学研究前沿：（1）基因组学：从草图到T2T、单倍型解析与泛基因组；（2）转录组与表观组学：时空分辨与染色质调控；（3）代谢组学与多组学整合：类黄酮、类胡萝卜素与苯丙烷代谢；（4）单细胞与空间组学新维度；（5）系统生物学应用：以黄龙病为例。全文引文均来自检索获得的文献并在参考文献中标注DOI；非柑橘特异性文献均明确标注「类比：非柑橘证据」。

## 2 基因组学前沿：从草图到T2T、单倍型解析与泛基因组

### 2.1 从参考基因组草图到高质量基因组

柑橘基因组学的起点可追溯至2013年甜橙参考基因组草图的发布[6]。该工作利用全基因组鸟枪法测序策略，结合柑橘基因组杂合度高的特点，构建了甜橙的首个参考基因组序列，为后续的基因注释、比较基因组与分子育种奠定了基础[6]。在早期阶段，由于测序读长与组装算法的限制，柑橘参考基因组普遍存在连续性不足、单倍型混杂等问题，难以支撑等位基因水平的精细研究[1,3]。随着第三代长读长测序与光学图谱、染色质构象捕获（Hi-C）等辅助组装技术的引入，柑橘基因组组装质量显著提升，多个物种与品种获得了染色体水平的高质量参考基因组[1,13,14]。

从历史上看，柑橘基因组资源的积累经历了从单一物种到多物种、从野生近缘种到栽培品种的扩展过程。Nakandala等（2025）系统回顾了柑橘基因组组装的过去、现在与未来，指出高质量的参考基因组是解析柑橘驯化、进化与遗传改良的核心资源[1]。Goldschmidt（2020）则在早期综述中总结了柑橘基因组计划的发展脉络与研究展望[3]。群体基因组学研究揭示了柑橘驯化的复杂杂交历史：Wu等（2014）通过对大量宽皮柑橘（Citrus reticulata Blanco）、柚（Citrus maxima (Burm.) Merr.）与橙类基因组的测序分析，提出栽培柑橘的驯化涉及广泛的种间杂交与渐渗，主要栽培类型多为杂交起源[4]。这一认识从根本上重塑了人们对柑橘分类与育种策略的理解，也凸显了单倍型水平基因组解析的必要性[4]。

### 2.2 单倍型解析与T2T基因组

柑橘基因组杂合度高，传统组装将两个单倍型混叠为一个"共识"序列，导致等位基因信息丢失。为此，研究者发展出单倍型解析（haplotype-resolved）组装策略，并进一步向端粒到端粒（T2T）组装迈进。2023年，一项研究报道了柑橘中单倍型解析的T2T组装，并开发了单倍型感知的注释流程（haplotype-aware annotation pipeline），对三个柑橘基因组进行了高质量重注释[8]。该工作展示了T2T组装在解析高度重复区域与完整等位基因结构方面的优势，为柑橘基因组学的精准化奠定了基础[8]。

在甜橙中，染色体水平的单倍型分型基因组为等位基因水平研究提供了范例：一项以黄龙病耐受性为案例的研究构建了甜橙的染色体水平单倍型分型基因组，实现了对耐病相关等位基因的精细解析[14]。另一项研究发布了巴西主栽甜橙品种'Pera IAC'（Citrus × sinensis 'Pera IAC'）的单倍型解析基因组，为该品种的遗传改良与全球甜橙比较基因组研究提供了重要资源[15]。在原始类群方面，单倍型解析的papeda（大翼橙类，柑橘属原始类群）基因组为探讨柑橘属的地理起源与演化提供了关键证据[13]。在宽皮柑橘中，温州蜜柑（Citrus unshiu Marcow.，亦称Citrus reticulata 'Unshiu'）的基因组起源研究进一步从基因组层面解析了这一无性系栽培品种的起源[16]。这些进展共同表明，单倍型解析与T2T组装已成为柑橘基因组学的新标准[1,8,13,14,15,16]。

**表1 柑橘基因组学代表性里程碑**

| 年份 | 代表工作 | 基因组类型 | 主要意义 | 文献 |
| --- | --- | --- | --- | --- |
| 2013 | 甜橙基因组草图 | 参考基因组草图 | 首个柑橘参考基因组 | [6] |
| 2014 | 宽皮柑橘/柚/橙群体基因组 | 群体基因组 | 揭示驯化复杂杂交历史 | [4] |
| 2023 | 三个柑橘基因组T2T组装 | 单倍型解析T2T | 柑橘T2T及注释流程 | [8] |
| 2023 | 甜橙染色体水平单倍型分型 | 单倍型分型 | 等位基因水平HLB研究 | [14] |
| 2024/2025 | papeda单倍型解析基因组 | 单倍型解析 | 柑橘地理起源与演化 | [13] |
| 2025 | 温州蜜柑基因组起源 | 基因组起源解析 | 无性系品种遗传结构 | [16] |
| 2026 | 'Pera IAC'甜橙单倍型基因组 | 单倍型解析 | 巴西主栽品种资源 | [15] |
| 2024/2025 | 栽培柑橘超级泛基因组 | 属级泛基因组 | 网状进化异地分化特征 | [7] |

### 2.3 泛基因组与超级泛基因组

单一参考基因组无法完全代表物种或属内的遗传多样性。泛基因组（pan-genome）通过整合多个个体的基因组序列，刻画物种的核心基因组（core genome）与可变基因组（dispensable/variable genome），已成为植物基因组学研究的前沿方向。在柑橘中，一项超级泛基因组（super-pangenome）研究整合了栽培柑橘及其野生近缘种的基因组资源，揭示了柑橘网状进化（reticulate evolution）异地分化（allopatric）阶段的重要进化特征，为理解柑橘驯化过程中基因家族的获得与丢失提供了系统框架[7]。该研究代表了柑橘泛基因组研究从"种内"向"属级"（包含近缘属）扩展的趋势[7]。

在遗传资源利用层面，基因组学正在加速柑橘育种的进程。有综述指出，基因组学手段能够系统挖掘柑橘种质资源库中的优良等位基因，将野生近缘种与地方品种中蕴含的抗病、抗逆与品质相关基因转化为育种可利用的遗传信息，从而缩短育种周期、提高选择效率[2]。结合群体基因组与泛基因组数据，柑橘育种有望从表型驱动的传统模式走向基因型驱动的精准设计育种[2,7]。需要说明的是，目前关于柑橘泛基因组核心文献的检索仍不充分（详见"局限与边界"），超级泛基因组研究是该领域为数不多的直接证据[7]。

## 3 转录组与表观组学：时空分辨与染色质调控

### 3.1 高时空分辨转录组与果实发育

果实发育与成熟是柑橘品质形成的核心过程，涉及细胞分裂、膨大、糖酸代谢、色素积累与芳香物质合成等一系列时序性事件，其转录调控高度动态且具有组织特异性。高通量转录组测序（RNA-seq）为此类研究提供了全局视角。一项高时空分辨率转录组研究系统刻画了甜橙果实发育与成熟过程中的基因表达动态，揭示了与糖积累、有机酸代谢、类胡萝卜素合成等品质相关通路的时间与空间表达模式[9]。

在组织分辨率层面，激光捕获显微切割（laser capture microdissection, LCM）技术实现了对柑橘果皮表皮与亚表皮组织的特异性转录组分析，为阐明果皮蜡质、精油与色素等性状的细胞类型特异性调控提供了经典范例[17]。这类"时空分辨"策略弥补了传统整果转录组在空间信息上的缺失，是连接转录组与果实形态建成的重要桥梁[9,17]。

此外，果实发育过程中的转录-翻译-蛋白水平调控并不总是同步。一项对两个甜橙品种果实发育与成熟期的比较转录组与蛋白质组研究，从转录与蛋白两个层面刻画了果实发育过程中的分子变化，提示多层面组学整合有助于更全面地理解果实品质形成机制[18]。

### 3.2 表观组学与染色质动态调控

表观遗传修饰是连接基因型与表型的重要调控层。在柑橘中，一项发表于The Plant Cell的研究系统解析了果实成熟过程中蔗糖与柠檬酸代谢的染色质动态调控程序，揭示了染色质可及性与组蛋白修饰的变化如何协同调控糖酸代谢关键基因的表达，从而影响果实的风味品质[10]。该研究将"表观组-转录组-代谢物"联系起来，为理解柑橘果实品质的调控网络提供了新视角[10]。

需要指出的是，柑橘表观组学研究目前仍处于起步阶段，ChIP-seq、ATAC-seq与Hi-C等技术的应用范围相对有限，相关证据多集中于果实发育与成熟阶段[10]。未来，将表观组学与群体遗传学、泛基因组相结合，有望揭示驯化过程中表观遗传变异的贡献（此为基于现有文献的展望性判断，尚缺乏直接证据）。

### 3.3 非生物胁迫响应的转录与多组学解析

柑橘产业正面临日益频繁的非生物胁迫，如干旱、高温、强光等，且这些胁迫常复合发生。转录组与多组学分析为解析柑橘的抗逆机制提供了有力工具。一篇综述系统总结了柑橘对多种非生物胁迫重叠应答的分子机制，并从遗传改良角度讨论了利用组学信息改良柑橘抗逆性的策略[19]。另一项多组学研究揭示了RNA翻译途径与未折叠蛋白反应（unfolded protein response, UPR）调控因子在柑橘耐受干旱、强光与高温复合胁迫中的潜在作用，提示翻译水平调控是非生物胁迫适应的关键环节[20]。这些研究共同表明，单一"静态"转录组难以全面反映胁迫响应，整合翻译组、蛋白组与代谢组的动态分析是未来方向[19,20]。



## 4 代谢组学与多组学整合：类黄酮、类胡萝卜素与苯丙烷代谢

### 4.1 苯丙烷类代谢

苯丙烷类化合物（phenylpropanoids）是柑橘中重要的生物活性物质，具有抗氧化、抗炎等保健功能，同时也是木质素、花青素等结构性或色素性物质的前体。一项柑橘代谢组学研究系统表征了果实中的苯丙烷类代谢物，结合转录组数据揭示了其生物合成与积累的组织特异性与发育动态，为挖掘柑橘功能性食品成分提供了代谢与分子层面的证据[21]。该研究凸显了代谢组学在"表型-代谢物-基因"链条中的桥梁作用[21]。

### 4.2 类黄酮代谢

类黄酮（flavonoids）是柑橘果实中含量丰富、结构多样的次生代谢物之一，其组成与含量直接影响果实的苦味、色泽与保健价值。利用一个宽皮柑橘×枳（Citrus reticulata Blanco × Poncirus trifoliata (L.) Raf.）遗传分离群体，研究者将基因组学与代谢组学相结合，在遗传群体水平上系统解析了柑橘类黄酮代谢的遗传基础[22]。这一"多组学+遗传群体"的策略代表了代谢组学研究从描述走向解析的重要范式[22]。

在类黄酮与类胡萝卜素共调控层面，一项对红肉脐橙'Cara cara'（Citrus sinensis (L.) Osbeck品种）的转录组与代谢组整合分析构建了类胡萝卜素与类黄酮生物合成的转录调控网络，揭示了两个代谢通路之间可能存在的协同调控机制[23]。此外，一项基于体细胞突变体的多组学分析发现，转录因子TCP7可等位调控多个类胡萝卜素合成基因，为果实色泽改良提供了新的分子靶点[24]。

### 4.3 果实品质与发育障碍的多组学解析

多组学整合在柑橘果实品质形成与发育障碍研究中展现出广泛应用。一项对橘橙（tangor，Citrus reticulata × Citrus sinensis）果实发育与品质形成的转录组-代谢组联合分析鉴定出一批与糖酸积累、风味形成相关的候选基因，为杂交柑橘品质育种提供了基因资源[25]。另一项工作利用转录组与代谢组技术解析了柑橘浮皮病（puffing disorder）的分子机制，发现果皮与果肉在代谢与基因表达层面的异常变化与该生理性病害密切相关[26]。这些案例说明，多组学整合能够在复杂农艺性状与生理障碍的解析中提供系统性证据[25,26]。

## 5 单细胞与空间组学新维度

### 5.1 单核转录组：细胞类型分辨率下的柑橘生物学

传统bulk转录组将整个组织或器官的细胞混合分析，掩盖了细胞类型间的异质性。单细胞/单核转录组技术可在单个细胞或细胞核水平解析基因表达，为植物生物学带来了全新维度。在柑橘中，一项单核转录组（single-nucleus RNA-seq, snRNA-seq）研究以粗柠檬（Citrus jambhiri Lush.）为材料，系统刻画了感染CLas后不同细胞类型的免疫应答图谱，展示了在细胞类型分辨率下解析病原-寄主互作的巨大潜力[11]。由于CLas是韧皮部限制性病原，这类细胞水平的应答信息对于理解抗性机制尤为重要[11]。

需要指出的是，目前柑橘单细胞组学研究仍非常有限，果实发育、花器官分化等生物学过程的单细胞图谱尚属空白（详见"局限与边界"）[11]。

### 5.2 空间组学：从组织到分子的空间定位

空间组学旨在将分子信息映射回其组织空间位置，是继单细胞组学之后的又一前沿方向。质谱成像（mass spectrometry imaging, MSI）技术可在不破坏组织的前提下原位检测代谢物、脂质与小分子的空间分布。Bjarnholt等（2014）系统阐述了植物代谢物质谱成像的原理与应用可能【类比：非柑橘证据】[27]；Wang等（2023）进一步总结了空间组学（含MSI）在植物研究中的最新进展与挑战【类比：非柑橘证据】[28]。这些方法学进展为柑橘果皮精油、类黄酮与类胡萝卜素的原位空间分布研究提供了技术基础，但截至目前，尚缺乏柑橘果实空间转录组与空间代谢组直接应用的高质量报道（详见"局限与边界"）[27,28]。

从技术路线看，将激光显微切割[17]、单核转录组[11]与质谱成像[27,28]等平台相整合，有望在柑橘中实现"单细胞-空间-代谢"的多维度解析，为果实品质与抗病性研究开辟新路径（此为基于方法学文献的展望，尚缺乏柑橘直接证据）。

## 6 系统生物学应用：以黄龙病（HLB）为例

### 6.1 HLB的多组学全景

黄龙病是由韧皮部限制性革兰氏阴性菌'Candidatus Liberibacter asiaticus'（CLas）引起的毁灭性柑橘病害，主要通过柑橘木虱（Diaphorina citri Kuwayama）传播，目前尚无根治性防治措施[5]。在分子互作层面，Hu等（2021）系统综述了柑橘与CLas之间的分子信号特征，涵盖效应蛋白、宿主免疫受体、代谢重编程与韧皮部运输等多个层面，为理解HLB的感病与抗病机制提供了整合框架[5]。

系统生物学视角下，HLB不仅仅是简单的病原侵染，而是一场复杂的宿主免疫失调事件。一项发表于Nature Communications的研究提出，HLB本质上是一种由病原触发的免疫性疾病——感染引发系统性的免疫过度激活，最终造成植株系统性衰退；该研究还发现抗氧化剂与赤霉素（gibberellin）处理可在一定程度上缓解症状[29]。这一概念框架深刻改变了人们对HLB病理机制的理解[29]。

### 6.2 多组学揭示的耐受机制

近年来，多项多组学研究从不同角度解析了柑橘对HLB的耐受机制。一项纵向多组学研究（转录组+蛋白质组+代谢组）追踪了甜橙在柑橘木虱接种CLas后的时序响应，揭示了感染早期与晚期的分子事件动态，为确定干预窗口提供了依据[12]。在品种比较层面，对耐病品种'Parson Brown'甜橙的多组学分析揭示了其独特的防御机制[30]。此外，一项覆盖枳（Poncirus trifoliata (L.) Raf.）、酸橘（Citrus sunki (Hayata) hort. ex Tanaka）、甜橙及其杂交后代的宽范围转录组分析，鉴定了与HLB耐受相关的候选基因与通路[31]；对感病与耐病品种早、晚期感染比较转录组的研究则进一步细化了不同遗传背景下免疫响应的差异[32]。这些结果与HLB作为病原触发免疫疾病的观点[29]相呼应，提示免疫稳态的维持可能是耐受性的关键（综合[29,30,31,32]的推断）。

在免疫调控网络层面，小RNA（sRNA）作为重要的免疫调控因子受到关注。一项比较转录组与小RNA组（sRNAome）联合分析表明，柑橘对HLB的免疫应答受到转录水平与转录后水平（小RNA介导）的协同调控[33]。此外，一项最新研究揭示了免疫调控节点基因NPR1在HLB中的作用：NPR1可抑制CLas诱导的胼胝质（callose）与活性氧（ROS）积累，从而在免疫稳态中发挥关键调控功能[34]。值得注意的是，NPR1相关功能在拟南芥等模式植物中已有大量积累，其在柑橘中的解析为"模式植物-作物"知识迁移提供了范例[34]。

**表2 HLB系统生物学代表性多组学研究**

| 研究层次 | 代表研究 | 技术路线 | 主要发现 | 文献 |
| --- | --- | --- | --- | --- |
| 纵向多组学 | 甜橙接种CLas时序响应 | 转录组+蛋白质组+代谢组 | 感染早晚期分子事件 | [12] |
| 品种比较 | 'Parson Brown'耐病甜橙 | 多组学 | 耐受性防御机制 | [30] |
| 跨物种比较 | 枳/酸橘/甜橙及杂交后代 | 宽范围转录组 | HLB耐受候选基因与通路 | [31] |
| 早晚期比较 | 感病与耐病品种 | 比较转录组 | 免疫响应遗传差异 | [32] |
| 转录后调控 | 感病柑橘 | 转录组+小RNA组 | 转录与转录后协同免疫 | [33] |
| 免疫节点 | NPR1功能解析 | 分子生理学 | 抑制胼胝质与ROS积累 | [34] |

### 6.3 从机制到防控策略

基于多组学认识，HLB防控策略正从传统的杀虫与清除病树向"免疫调控+精准诊断"方向演进。蛋白质组学研究揭示，热疗（thermotherapy）可通过诱导宿主特定分子机制抑制CLas增殖，为物理防治提供了分子依据[35]。在区域层面，HLB对拉丁美洲等主要柑橘产区的持续威胁提示，全球柑橘产业亟需将组学研究成果转化为可推广的综合防控方案[36]。在检测层面，基因组学与分子诊断技术的结合显著提升了柑橘病害早期检测的灵敏度与通量，为田间监测与检疫提供了支撑[37]。总体而言，HLB研究已成为柑橘系统生物学应用最集中的领域，其"病原-寄主-媒介-环境"多维度互作的组学解析，为其他果树病害研究提供了范式[5,12,29,30,31,32,33,34,35,36,37]。

## 7 结论与展望

过去十余年，柑橘组学研究完成了从"单组学描述"到"多组学整合"、从"参考基因组"到"单倍型解析/T2T/泛基因组"、从"bulk水平"到"单细胞与空间水平"的三重跃迁[1,7,8,9,11]。基因组学层面，单倍型解析与T2T组装已成为新标准，超级泛基因组正在拓展对柑橘网状进化与驯化历史的认识[1,7,8,13,14,15,16]；转录组与表观组学层面，时空分辨技术与染色质动态调控研究深化了人们对果实品质形成机制的理解[9,10,17]；代谢组学与多组学整合层面，类黄酮、类胡萝卜素与苯丙烷代谢的遗传解析为品质育种提供了基因资源[21,22,23,24]；单细胞与空间组学层面，snRNA-seq已初步展现其在细胞类型分辨率下解析生物学问题的能力[11]；系统生物学层面，以HLB为代表的研究正在推动从"治病"向"免疫调控"的范式转变[29,34]。

展望未来，柑橘组学研究将呈现以下趋势：（1）更多物种与品种的T2T与单倍型解析基因组将陆续发布，属级泛基因组（超级泛基因组）资源将更加完善[1,7]；（2）单细胞与空间组学技术将逐步覆盖果实发育、逆境响应与病原互作等过程，填补当前证据空白[11,27,28]；（3）多组学整合与人工智能（AI）方法相结合，将提升复杂性状因果基因的预测与验证效率[2,18]；（4）"组学-育种-栽培-植保"的贯通将加速柑橘产业的可持续发展[2,29]。

## 8 局限与边界

本文综述基于给定的检索文献，存在以下局限与信息缺口：

（1）柑橘泛基因组核心文献检索不足：目前仅有超级泛基因组研究1篇（DOI: 10.1111/pbi.70553）作为直接证据，尚难以全面评述柑橘泛基因组的核心基因集、结构变异与驯化相关变异[7]。

（2）柑橘果实发育单细胞图谱缺失：现有snRNA-seq研究聚焦于HLB免疫响应（DOI: 10.1093/hr/uhaf265），果实发育、花发育等过程的单细胞转录组证据不足[11]。

（3）柑橘空间转录组与空间代谢组的直接应用证据空缺：本文仅能引用植物整体方法学综述（DOI: 10.1039/c3np70100j、DOI: 10.3389/fpls.2023.1273010）作为方法学参照，两者均属非柑橘证据【类比：非柑橘证据】[27,28]。

（4）柑橘蛋白质组学系统综述缺乏：现有蛋白组学证据分散于果实发育（DOI: 10.1186/s12864-017-4366-2）与HLB热疗（DOI: 10.1186/s12870-016-0942-x）等具体研究方向[18,35]。

（5）部分检索条目元数据不完整（作者、年份缺失），引用时已在参考文献中注明；个别条目未提供DOI（如2016年"Omics studies of citrus, grape and rosaceae fruit trees"及2017年Satsuma基因组草案），为避免无法溯源的引用，本文未将其纳入参考文献列表。

（6）本文未对检索文献进行偏倚校正，结论主要反映检索所得文献的共识，未来可结合更全面的数据库检索进行系统评价。

## 参考文献

[1] Nakandala U, Furtado A, Henry RJ. Citrus genomes: past, present and future. Horticulture Research, 2025. DOI: 10.1093/hr/uhaf033

[2] Genomics unlocks the potential of genetic resources for citrus breeding. Breeding Science, 2024. DOI: 10.1270/jsbbs.24047

[3] Goldschmidt EE. The Citrus Genome: Past, Present and Future. Compendium of Plant Genomes, 2020. DOI: 10.1007/978-3-030-15308-3_1

[4] Wu GA, et al. Sequencing of diverse mandarin, pummelo and orange genomes reveals complex history of admixture during citrus domestication. Nature Biotechnology, 2014. DOI: 10.1038/nbt.2906

[5] Hu B, Rao MJ, Deng X, Pandey SS. Molecular signatures between citrus and Candidatus Liberibacter asiaticus. PLOS Pathogens, 2021. DOI: 10.1371/journal.ppat.1010071

[6] Xu Q, et al. The draft genome of sweet orange (Citrus sinensis). Nature Genetics, 2013. DOI: 10.1038/ng.2472

[7] A Super-Pangenome for Cultivated Citrus Reveals Evolutive Features During the Allopatric Phase of Their Reticulate Evolution. Plant Biotechnology Journal, 2024/2025. DOI: 10.1111/pbi.70553

[8] Haplotype-resolved telomere-to-telomere assembly and haplotype-aware annotation pipeline enable high-quality reannotation of three Citrus genomes. Horticulture Research, 2023. DOI: 10.1093/hr/uhag048

[9] High-spatiotemporal-resolution transcriptomes provide insights into fruit development and ripening in Citrus sinensis. Plant Biotechnology Journal, 2022. DOI: 10.1111/pbi.13549

[10] Song X, Wang TT, et al. Dynamic chromatin regulatory programs of sucrose and citric acid metabolism during fruit ripening in Citrus. The Plant Cell, 2023. DOI: 10.1093/plcell/koag060

[11] Single-nucleus transcriptomics reveals the cellular immune responses to Candidatus Liberibacter asiaticus in rough lemon. Horticulture Research, 2025. DOI: 10.1093/hr/uhaf265

[12] Longitudinal Transcriptomic, Proteomic, and Metabolomic Response of Citrus sinensis to Diaphorina citri Inoculation of Candidatus Liberibacter asiaticus. Journal of Proteome Research, 2023. DOI: 10.1021/acs.jproteome.3c00485

[13] Haplotype-resolved genome of a papeda provides insights into the geographical origin and evolution of Citrus. Journal of Integrative Plant Biology, 2024/2025. DOI: 10.1111/jipb.13819

[14] A chromosome-level phased genome enabling allele-level studies in sweet orange: a case study on citrus Huanglongbing tolerance. Horticulture Research, 2023. DOI: 10.1093/hr/uhac247

[15] Haplotype-resolved genome of Citrus × sinensis 'Pera IAC', the most widely cultivated sweet orange in Brazil. Scientific Data, 2026（在线2025）. DOI: 10.1038/s41597-026-07229-9

[16] Genomic origin of Citrus reticulata 'Unshiu'. Horticulture Research, 2025. DOI: 10.1093/hr/uhaf015

[17] Matas AJ, Agustí J, Tadeo FR, Talón M. Tissue-specific transcriptome profiling of the citrus fruit epidermis and subepidermis using laser capture microdissection. Journal of Experimental Botany, 2010. DOI: 10.1093/jxb/erq153

[18] Comparative transcriptome and proteome profiling of two Citrus sinensis cultivars during fruit development and ripening. BMC Genomics, 2017. DOI: 10.1186/s12864-017-4366-2（作者信息待补充）

[19] Overlapping responses to multiple abiotic stresses in citrus: from mechanism understanding to genetic improvement. Fruit Research, 2023. DOI: 10.1007/s44281-023-00007-2

[20] Omics analyses in citrus reveal a possible role of RNA translation pathways and Unfolded Protein Response regulators in the tolerance to combined drought, high irradiance, and heat stress. Horticulture Research, 2023. DOI: 10.1093/hr/uhad107

[21] A metabolomics study in citrus provides insight into bioactive phenylpropanoid metabolism. Horticulture Research, 2023. DOI: 10.1093/hr/uhad267

[22] Multiomics-based dissection of citrus flavonoid metabolism using a Citrus reticulata × Poncirus trifoliata population. Horticulture Research, 2021. DOI: 10.1038/s41438-021-00472-8

[23] Integrated Transcriptomic and Metabolomic analysis reveals a transcriptional regulation network for the biosynthesis of carotenoids and flavonoids in 'Cara cara' navel Orange. BMC Plant Biology, 2020. DOI: 10.1186/s12870-020-02808-3

[24] Multi-omics analysis of somatic mutants reveals TCP7 allelically regulates multiple carotenogenic genes in citrus. 2026（在线2025）. DOI: 10.1186/s43897-025-00193-9（作者信息待补充）

[25] Combined Transcriptome and Metabolome Analyses Reveal Candidate Genes Involved in Tangor (Citrus reticulata × Citrus sinensis) Fruit Development and Quality Formation. International Journal of Molecular Sciences, 2022. DOI: 10.3390/ijms23105457

[26] Transcriptome and metabolome analysis of Citrus fruit to elucidate puffing disorder. Plant Science, 2014. DOI: 10.1016/j.plantsci.2013.12.003

[27] Bjarnholt N, Li B, D'Alvise J, Janfelt C. Mass spectrometry imaging of plant metabolites – principles and possibilities. Natural Product Reports, 2014. DOI: 10.1039/c3np70100j【类比：非柑橘证据】

[28] Wang X, Han J, Li Z, Li B. Insight into plant spatial omics: mass spectrometry imaging. Frontiers in Plant Science, 2023. DOI: 10.3389/fpls.2023.1273010【类比：非柑橘证据】

[29] Citrus Huanglongbing is a pathogen-triggered immune disease that can be mitigated with antioxidants and gibberellin. Nature Communications, 2022. DOI: 10.1038/s41467-022-28189-9

[30] Multi-omics analyses reveal the defense mechanisms behind the tolerance of the 'Parson Brown' sweet orange to Huanglongbing. BMC Plant Biology, 2025. DOI: 10.1186/s12870-025-07372-2

[31] Wide-ranging transcriptomic analysis of Poncirus trifoliata, Citrus sunki, Citrus sinensis and contrasting hybrids reveals HLB tolerance mechanisms. Scientific Reports, 2020. DOI: 10.1038/s41598-020-77840-2

[32] Comparative transcriptome profiling of susceptible and tolerant citrus species at early and late stage of infection by 'Candidatus Liberibacter asiaticus'. Frontiers in Plant Science, 2023. DOI: 10.3389/fpls.2023.1191029

[33] Comparative Transcriptome and sRNAome Analysis Suggest Coordinated Citrus Immune Responses against Huanglongbing Disease. Plants, 2024. DOI: 10.3390/plants13111496

[34] NPR1 suppresses Candidatus Liberibacter asiaticus-induced callose and reactive oxygen species accumulation. Plant Physiology, 2025. DOI: 10.1093/plphys/kiaf532

[35] Proteomics analysis reveals novel host molecular mechanisms associated with thermotherapy of 'Ca. Liberibacter asiaticus'-infected citrus plants. BMC Plant Biology, 2016. DOI: 10.1186/s12870-016-0942-x

[36] Huanglongbing as a Persistent Threat to Citriculture in Latin America. Biology, 2025. DOI: 10.3390/biology14040335（作者信息待补充）

[37] Genomic Analysis for Citrus Disease Detection. OBM Genetics, 2021. DOI: 10.21926/obm.genet.2101124（作者信息待补充）
