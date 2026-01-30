import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ===================== 解决Matplotlib中文乱码问题 =====================
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号

# ===================== 模型加载（添加异常处理） =====================
try:
    model = joblib.load('CatBoost.pkl')  # 加载训练好的CatBoost模型
except FileNotFoundError:
    st.error("错误：未找到模型文件 CatBoost.pkl，请检查文件路径是否正确！")
    st.stop()  # 终止程序运行
except Exception as e:
    st.error(f"模型加载失败：{str(e)}，请检查模型文件是否损坏！")
    st.stop()

# ===================== 定义特征选项（优化冗余，通用字典复用） =====================
# 性别选项
GENDER_options = {1: '男生', 2: '女生'}
# 心理状态通用选项（所有D开头的心理特征复用，无需重复定义）
PSYCH_OPTIONS = {
    1: '没有或偶尔',  2: '有时',  3: '时常或一半时间',
    4: '多数时间或持续', 5: '不清楚'
}
# 每周体育课节数
PEC_options = {1: '0节', 2: '1节', 3: '2节',4: '3节',5: '4节', 6: '5节及以上'}
# 吸烟饮酒史
SACH_options = {0: '无', 1: '有'}

# ===================== Streamlit 页面UI =====================
st.title("学生1年后肥胖风险预测")
# 侧边栏输入样本数据
st.sidebar.header("请输入学生信息")

# 按照模型训练时的特征顺序收集输入（务必与训练时特征顺序一致！）
Q7_all = st.sidebar.number_input("体重（kg）:", min_value=20.0, max_value=150.0, value=50.0, step=0.5)
Q6_all = st.sidebar.number_input("身高（cm）:", min_value=100.0, max_value=220.0, value=160.0, step=1.0)
AGE = st.sidebar.number_input("年龄（岁）:", min_value=6, max_value=20, value=12, step=1)
GENDER = st.sidebar.selectbox("性别:", options=list(GENDER_options.keys()), format_func=lambda x: GENDER_options[x])
D16 = st.sidebar.selectbox("我过着幸福的生活:", options=list(PSYCH_OPTIONS.keys()), format_func=lambda x: PSYCH_OPTIONS[x])
D10 = st.sidebar.selectbox("我感到恐惧:", options=list(PSYCH_OPTIONS.keys()), format_func=lambda x: PSYCH_OPTIONS[x])
PEC = st.sidebar.selectbox("每周体育课节数:", options=list(PEC_options.keys()), format_func=lambda x: PEC_options[x])
D12 = st.sidebar.selectbox("我很幸福:", options=list(PSYCH_OPTIONS.keys()), format_func=lambda x: PSYCH_OPTIONS[x])
D2 = st.sidebar.selectbox("我不想吃东西:", options=list(PSYCH_OPTIONS.keys()), format_func=lambda x: PSYCH_OPTIONS[x])
D17 = st.sidebar.selectbox("我曾经放声痛哭:", options=list(PSYCH_OPTIONS.keys()), format_func=lambda x: PSYCH_OPTIONS[x])
DST = st.sidebar.selectbox("每日睡眠时长（小时）:", options=[6,7,8,9,10,11,12], format_func=lambda x: f"{x}小时")
D1 = st.sidebar.selectbox("以前从不困扰我的事情现在让我烦恼:", options=list(PSYCH_OPTIONS.keys()), format_func=lambda x: PSYCH_OPTIONS[x])
SACH = st.sidebar.selectbox("是否有过吸烟饮酒史：", options=list(SACH_options.keys()), format_func=lambda x: SACH_options[x])
D8 = st.sidebar.selectbox("我觉得未来有希望:", options=list(PSYCH_OPTIONS.keys()), format_func=lambda x: PSYCH_OPTIONS[x])
D7 = st.sidebar.selectbox("我感到做什么事都很费力:", options=list(PSYCH_OPTIONS.keys()), format_func=lambda x: PSYCH_OPTIONS[x])
D18 = st.sidebar.selectbox("我感到忧愁:", options=list(PSYCH_OPTIONS.keys()), format_func=lambda x: PSYCH_OPTIONS[x])

# ===================== 特征数据处理与预测 =====================
# 按模型训练特征顺序整理输入（务必与训练时一致！）
feature_values = [Q7_all,Q6_all,AGE,GENDER, D16, D10, PEC,D12,D2,D17,DST,D1,SACH,D8,D7,D18]
# 定义特征名（请替换为【模型训练时的实际特征名】，顺序与feature_values严格对应）
feature_names = ['体重(kg)', '身高(cm)', '年龄', '性别', 'D16', 'D10', '体育课节数', 'D12', 'D2', 'D17', '睡眠时长', 'D1', '吸烟饮酒史', 'D8', 'D7', 'D18']
# 转换为DataFrame（模型预测更规范，避免数组格式问题）
features = pd.DataFrame([feature_values], columns=feature_names)

# 预测按钮逻辑
if st.button("开始预测"):
    try:
        # 预测类别和概率
        predicted_class = model.predict(features)[0]  # 0=低风险，1=高风险
        predicted_proba = model.predict_proba(features)[0]  # [非肥胖概率, 肥胖概率]

        # 展示预测结果
        st.write(f"### 📊 预测结果")
        st.write(f"**结论:** {'1年后肥胖风险高' if predicted_class == 1 else '1年后肥胖风险低'}")
        
        # 计算对应类别的概率（百分比）
        probability = predicted_proba[predicted_class] * 100
        # 生成个性化提示
        if predicted_class == 1:
            advice = f"根据模型预测，该学生1年后的肥胖风险较高，风险概率为 **{probability:.1f}%**，建议关注饮食与运动习惯！"
        else:
            advice = f"根据模型预测，该学生1年后的肥胖风险较低，非肥胖概率为 **{probability:.1f}%**，请继续保持良好习惯！"
        st.success(advice)

        # ===================== 预测概率可视化 =====================
        st.write(f"### 📈 预测概率分布")
        prob_data = {'非肥胖': predicted_proba[0], '肥胖': predicted_proba[1]}
        plt.figure(figsize=(10, 3))  # 设置图表大小
        # 绘制水平条形图，自定义颜色
        bars = plt.barh(['非肥胖', '肥胖'], [prob_data['非肥胖'], prob_data['肥胖']], color=['#4CAF50', '#F44336'])
        # 图表样式设置
        plt.title("1年后肥胖风险预测概率分布", fontsize=16, fontweight='bold')
        plt.xlabel("概率值", fontsize=12, fontweight='bold')
        plt.ylabel("风险类别", fontsize=12, fontweight='bold')
        plt.xlim(0, 1)  # 概率轴范围0-1
        # 隐藏顶部、右侧边框，让图表更简洁
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        # 为条形图添加概率数值标签
        for i, v in enumerate([prob_data['非肥胖'], prob_data['肥胖']]):
            plt.text(v + 0.01, i, f"{v:.2f}", va='center', fontsize=12, fontweight='bold')
        # 在Streamlit中展示图表
        st.pyplot(plt)

    except Exception as e:
        st.error(f"预测过程中出现错误: {str(e)}")
        st.info("请检查：1.模型文件是否为训练好的CatBoost模型；2.输入数据是否符合范围；3.特征顺序是否与模型训练时一致。")
