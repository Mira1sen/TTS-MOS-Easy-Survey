#demo FYI
import gradio as gr
import os
import random

class MOSApp:
    MOS = {
        1: "1-Bad", 1.5: "1.5", 2: "2-Poor", 2.5: "2.5", 3: "3-Fair",
        3.5: "3.5", 4: "4-Good", 4.5: "4.5", 5: "5-Excellent"
    }

    def __init__(self):
        random.seed(10)
        self.prompt_floder = "prompt"
        
        # 动态检测模型文件夹
        model_folders = [folder for folder in os.listdir('.') if os.path.isdir(folder) and folder != self.prompt_floder and not folder.startswith(".")]
        self.model_folders = sorted(model_folders)
        
        # 获取每个模型文件夹下的音频文件
        self.model_files = {}
        for folder in self.model_folders:
            self.model_files[folder] = sorted([os.path.join(folder, file) for file in os.listdir(folder)])
        
        # 获取prompt文件夹下的音频文件
        self.prompt_files = sorted([os.path.join(self.prompt_floder, file) for file in os.listdir(self.prompt_floder)])
        
        # 确保所有模型文件夹的音频数量与prompt一致
        min_length = min(len(self.prompt_files), *(len(files) for files in self.model_files.values()))
        self.prompt_files = self.prompt_files[:min_length]
        for folder in self.model_folders:
            self.model_files[folder] = self.model_files[folder][:min_length]
        
        # 为每个音频组创建随机顺序
        self.audio_order = []
        for i in range(len(self.prompt_files)):
            models = [(self.model_files[folder][i], folder, idx) for idx, folder in enumerate(self.model_folders, start=1)]
            random.shuffle(models)
            self.audio_order.append(models)

        print(self.audio_order)

        # 添加已使用ID的集合
        self.used_ids = set()
        # 如果results.csv存在，读取所有已使用的ID
        if os.path.exists('results.csv'):
            try:
                with open('results.csv', 'r') as f:
                    # 跳过表头
                    next(f, None)
                    for line in f:
                        if line.strip():
                            # 获取每行的第一个字段（ID）
                            used_id = line.split(',')[0]
                            self.used_ids.add(used_id)
            except Exception as e:
                print(f"读取results.csv时出错: {e}")

    def initialize_state(self):
        return {
            "index": 0,
            "selected_MOS": {folder: [] for folder in self.model_folders},  # 各模型的评分
            "tester_id": "",
            "data_store": {}
        }

    def submit_options(self, state, *options):
        if not state["tester_id"]:
            return (
                *([None] * (1 + len(self.model_folders))),
                "请先输入您的测试者ID",
                *([0.5] * len(self.model_folders)),
                state
            )
        
        if state["index"] >= len(self.prompt_files):
            return (
                *([None] * (1 + len(self.model_folders))),
                "## 测评已经结束！感谢您的反馈！\n## ",
                *([0.5] * len(self.model_folders)),
                state
            )
        
        # 检查所有评分是否都已选择
        if 0.5 in options:
            current_audios = self.audio_order[state["index"]]
            return (
                *(audio[0] for audio in current_audios),  # 各模型的音频
                self.prompt_files[state["index"]],
                "#### 无效提交！请为所有音频选择评分后再提交",
                *([0.5] * len(self.model_folders)),
                state
            )

        # 根据随机顺序记录评分
        current_order = self.audio_order[state["index"]]
        for i, (_, folder, _) in enumerate(current_order):
            state["selected_MOS"][folder].append(options[i])

        state["index"] += 1
        
        if state["index"] < len(self.prompt_files):
            next_audios = self.audio_order[state["index"]]
            return (
                *(audio[0] for audio in next_audios),  # 各模型的音频
                self.prompt_files[state["index"]],
                f"#### 您正在评价第 {state['index']+1} 组音频，共 {str(len(self.prompt_files))} 组。提交后请向上滚动收听新的音频",
                *([0.5] * len(self.model_folders)),
                state
            )
        else:
            # 保存所有评分结果为CSV格式
            file_exists = os.path.isfile('results.csv')
            
            with open('results.csv', 'a') as f:
                if not file_exists:
                    header = "id,model," + ",".join([f"MOS{i+1}" for i in range(len(self.prompt_files))])
                    f.write(header + "\n")
                
                tester_id = state["tester_id"]
                for folder in self.model_folders:
                    scores = ",".join(map(str, state["selected_MOS"][folder]))
                    f.write(f"{tester_id},{folder},{scores}\n")
            
            return (
                *([None] * (1 + len(self.model_folders))),
                "## 感谢您的反馈！测评结束",
                *([0.5] * len(self.model_folders)),
                state
            )

    def set_tester_id(self, id, state):
        if not id:
            return (
                "## 请输入有效的ID！", 
                state, 
                *([None] * (1 + len(self.model_folders))),
                ""
            )
        
        if id in self.used_ids:
            return (
                "## 该ID已被使用，请使用新的ID！", 
                state, 
                *([None] * (1 + len(self.model_folders))),
                ""
            )
        
        state["tester_id"] = id
        for folder in self.model_folders:
            state["selected_MOS"][folder] = []
        state["index"] = 0
        self.used_ids.add(id)
        
        first_audios = self.audio_order[0]

        return (
            f"## 您的ID: {state['tester_id']}", 
            state, 
            *(audio[0] for audio in first_audios),
            self.prompt_files[0],
            f"#### 您正在评价第 {state['index']+1} 组音频，共 {str(len(self.prompt_files))} 个。提交后请向上滚动收听新的音频"
        )

    def create_interface(self):
        with gr.Blocks() as demo:
            state = gr.State(self.initialize_state())

            gr.Markdown("# TTS MOS Survey System")

            gr.Markdown("## 首先请输入您的测试者ID")
            
            with gr.Row():
                tester_id_input = gr.Textbox(label="输入测试者ID")
                set_id_button = gr.Button("确认ID")
                id_display = gr.Markdown()
            
            gr.Markdown("------")
            gr.Markdown("## 根据您听到的音频回答以下问题")
            score_description = gr.Markdown("""
                ### 1. 请从自然度、清晰度和发音准确度方面评价语音的整体质量
                | 分数 | 自然度/人声相似度 | 机器音特征 |
                |-------|---------------------|-------------|
                | 5 优秀 | 完全自然的语音 | 无法察觉机器音特征 |
                | 4 良好 | 大部分自然的语音 | 可以察觉但不影响听感 |
                | 3 一般 | 自然与不自然程度相当 | 明显可察觉且略有影响 |
                | 2 较差 | 大部分不自然的语音 | 令人不适但尚可接受 |
                | 1 很差 | 完全不自然的语音 | 非常明显且无法接受 |
                """)
            gr.Markdown("------")
            gr.Markdown("## 请仔细听以下prompt音频：")
            with gr.Row():
                prompt_audio = gr.Audio(
                    value=None,
                    type='filepath',
                    label="Prompt音频"
                )

            gr.Markdown("------")
            gr.Markdown("## 请仔细听以下不同模型的音频：")
            
            audio_elements = []
            option_elements = []
            label_elements = []
            for i, folder in enumerate(self.model_folders):
                gr.Markdown(f"### model {i+1}：")
                audio = gr.Audio(None, type='filepath')
                option = gr.Slider(minimum=0.5, maximum=5, step=0.5,
                                  value=0.5, container=False, interactive=True)
                label = gr.HTML(self.get_slider_labels_html())
                audio_elements.append(audio)
                option_elements.append(option)  # 保存滑块实例
            
            with gr.Row():
                submit = gr.Button("提交")
            current_file = gr.Markdown("#### 请先输入ID再开始评测")

            set_id_button.click(
                self.set_tester_id, 
                inputs=[tester_id_input, state], 
                outputs=[id_display, state, *audio_elements, prompt_audio, current_file]
            )
            submit.click(
                self.submit_options, 
                inputs=[state, *option_elements], 
                outputs=[*audio_elements, prompt_audio, current_file, *option_elements, state]
            )

        return demo

    @staticmethod
    def get_slider_labels_html():
        return """
        <style>
            .slider-labels {
                display: flex;
                justify-content: space-between;
                margin-top: -10px;
                font-size: 12px;
            }
            .slider-labels div {
                text-align: center;
                width: 5%;
            }
            .slider-labels div:first-child {
                text-align: left;
            }
            .slider-labels div:last-child {
                text-align: right;
            }
        </style>
        <div class="slider-labels">
            <div>请选择</div>
            <div>1 很差</div>
            <div>1.5</div>
            <div>2 较差</div>
            <div>2.5</div>
            <div>3 一般</div>
            <div>3.5</div>
            <div>4 良好</div>
            <div>4.5</div>
            <div>5 优秀</div>
        </div>
        """

if __name__ == "__main__":
    app = MOSApp()
    demo = app.create_interface()
    demo.launch(server_name="0.0.0.0", server_port=8565)