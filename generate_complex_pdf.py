"""
generate_complex_pdf.py - 生成包含表格、图片的复杂 PDF 测试文档
用于测试 RAG 系统对复杂 PDF 的解析和分片能力
"""
import os
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm, mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.lib.colors import HexColor, black, white, grey
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Image, KeepTogether
)
from reportlab.platypus.flowables import HRFlowable
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.graphics.shapes import Drawing, Rect, String, Line, Circle, Polygon, Group
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.charts.linecharts import HorizontalLineChart

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "documents"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

_fonts_registered = False


def register_fonts():
    global _fonts_registered
    if _fonts_registered:
        return
    font_dirs = [
        "C:/Windows/Fonts",
        "C:/WinNT/Fonts",
    ]
    for fd in font_dirs:
        if not Path(fd).exists():
            continue
        font_map = {
            "SimHei": "simhei.ttf",
            "SimSun": "simsun.ttc",
            "SimKai": "simkai.ttf",
            "SimFang": "simfang.ttf",
            "Microsoft YaHei": "msyh.ttc",
            "Microsoft YaHei Bold": "msyhbd.ttc",
        }
        for name, filename in font_map.items():
            fp = Path(fd) / filename
            if fp.exists():
                try:
                    pdfmetrics.registerFont(TTFont(name, str(fp)))
                except Exception:
                    pass
    _fonts_registered = True


def get_styles():
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        name="CNTitle", fontName="SimHei", fontSize=22,
        leading=30, alignment=TA_CENTER, spaceAfter=6 * mm,
        textColor=HexColor("#1a1a2e"),
    ))
    styles.add(ParagraphStyle(
        name="CNSubtitle", fontName="Microsoft YaHei", fontSize=12,
        leading=18, alignment=TA_CENTER, spaceAfter=10 * mm,
        textColor=HexColor("#666666"),
    ))
    styles.add(ParagraphStyle(
        name="CNH1", fontName="SimHei", fontSize=16,
        leading=24, spaceBefore=10 * mm, spaceAfter=4 * mm,
        textColor=HexColor("#16213e"),
    ))
    styles.add(ParagraphStyle(
        name="CNH2", fontName="Microsoft YaHei Bold", fontSize=13,
        leading=20, spaceBefore=6 * mm, spaceAfter=3 * mm,
        textColor=HexColor("#0f3460"),
    ))
    styles.add(ParagraphStyle(
        name="CNH3", fontName="Microsoft YaHei Bold", fontSize=11,
        leading=17, spaceBefore=4 * mm, spaceAfter=2 * mm,
        textColor=HexColor("#533483"),
    ))
    styles.add(ParagraphStyle(
        name="CNBody", fontName="SimSun", fontSize=10,
        leading=18, alignment=TA_JUSTIFY, spaceAfter=3 * mm,
        firstLineIndent=2 * 10,
    ))
    styles.add(ParagraphStyle(
        name="CNBodyNoIndent", fontName="SimSun", fontSize=10,
        leading=18, alignment=TA_JUSTIFY, spaceAfter=3 * mm,
    ))
    styles.add(ParagraphStyle(
        name="CNNote", fontName="SimKai", fontSize=9,
        leading=15, textColor=HexColor("#888888"),
        spaceAfter=2 * mm, leftIndent=5 * mm,
    ))
    styles.add(ParagraphStyle(
        name="CNTableCell", fontName="SimSun", fontSize=9,
        leading=14, alignment=TA_CENTER,
    ))
    styles.add(ParagraphStyle(
        name="CNTableHeader", fontName="SimHei", fontSize=9,
        leading=14, alignment=TA_CENTER, textColor=white,
    ))
    styles.add(ParagraphStyle(
        name="CNTableLeft", fontName="SimSun", fontSize=9,
        leading=14, alignment=TA_LEFT,
    ))
    return styles


def make_horizontal_line():
    return HRFlowable(width="100%", thickness=0.5, color=HexColor("#cccccc"),
                       spaceBefore=3 * mm, spaceAfter=3 * mm)


def build_document():
    register_fonts()
    styles = get_styles()
    output_path = str(OUTPUT_DIR / "智能巡检机器人技术手册.pdf")

    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        leftMargin=2.5 * cm, rightMargin=2.5 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm,
        title="智能巡检机器人技术手册",
        author="技术研发部",
    )

    story = []

    # ============================================================
    # 封面
    # ============================================================
    story.append(Spacer(1, 4 * cm))
    story.append(Paragraph("智能巡检机器人", styles["CNTitle"]))
    story.append(Paragraph("XJ-8000 系列技术手册", styles["CNTitle"]))
    story.append(Spacer(1, 1 * cm))
    story.append(Paragraph("版本 V3.2.1  |  机密等级：内部", styles["CNSubtitle"]))
    story.append(Paragraph("技术研发部编制  |  2025年12月", styles["CNSubtitle"]))
    story.append(Spacer(1, 2 * cm))

    # 封面上的简单图示
    d = Drawing(400, 120)
    d.add(Rect(50, 10, 300, 100, fillColor=HexColor("#e8f4f8"), strokeColor=HexColor("#16213e"), strokeWidth=1.5))
    d.add(Circle(200, 60, 30, fillColor=HexColor("#0f3460"), strokeColor=HexColor("#16213e"), strokeWidth=1))
    d.add(String(185, 55, "XJ-8000", fontName="SimHei", fontSize=10, fillColor=white))
    d.add(Line(80, 40, 140, 40, strokeColor=HexColor("#533483"), strokeWidth=2))
    d.add(Line(260, 40, 320, 40, strokeColor=HexColor("#533483"), strokeWidth=2))
    d.add(Circle(80, 40, 4, fillColor=HexColor("#e94560")))
    d.add(Circle(320, 40, 4, fillColor=HexColor("#e94560")))
    story.append(d)
    story.append(PageBreak())

    # ============================================================
    # 目录页
    # ============================================================
    story.append(Paragraph("目  录", styles["CNTitle"]))
    story.append(Spacer(1, 8 * mm))
    toc_items = [
        ("1", "产品概述", "3"),
        ("1.1", "产品简介", "3"),
        ("1.2", "应用场景", "3"),
        ("1.3", "核心优势", "4"),
        ("2", "技术参数", "5"),
        ("2.1", "硬件规格", "5"),
        ("2.2", "传感器配置", "6"),
        ("2.3", "通信与接口", "7"),
        ("3", "性能测试数据", "8"),
        ("3.1", "续航与充电", "8"),
        ("3.2", "检测精度", "9"),
        ("3.3", "环境适应性", "10"),
        ("4", "系统架构", "11"),
        ("5", "部署方案对比", "13"),
        ("6", "维护保养指南", "15"),
        ("7", "故障排查", "17"),
    ]
    toc_data = [[Paragraph("<b>章节</b>", styles["CNTableHeader"]),
                 Paragraph("<b>内容</b>", styles["CNTableHeader"]),
                 Paragraph("<b>页码</b>", styles["CNTableHeader"])]]
    for num, title, page in toc_items:
        toc_data.append([
            Paragraph(num, styles["CNTableCell"]),
            Paragraph(title, styles["CNTableLeft"]),
            Paragraph(page, styles["CNTableCell"]),
        ])
    toc_table = Table(toc_data, colWidths=[1.5 * cm, 10 * cm, 2 * cm])
    toc_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#16213e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), white),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#cccccc")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, HexColor("#f5f7fa")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(toc_table)
    story.append(PageBreak())

    # ============================================================
    # 第1章：产品概述
    # ============================================================
    story.append(Paragraph("1  产品概述", styles["CNH1"]))
    story.append(make_horizontal_line())

    story.append(Paragraph("1.1  产品简介", styles["CNH2"]))
    story.append(Paragraph(
        "XJ-8000 系列智能巡检机器人是本公司自主研发的第三代工业级自主巡检平台，"
        "集成了多模态感知、自主导航、边缘计算和远程协作等核心技术。该机器人能够在"
        "复杂工业环境中实现全天候、全自主的巡检作业，广泛应用于电力变电站、石化管线、"
        "数据中心机房、矿山巷道等场景。机器人本体采用 IP67 防护等级设计，可在 -30℃ "
        "至 +65℃ 的极端温度环境下稳定运行。整机重量 42kg，最大爬坡角度 30°，"
        "具备越障能力，可跨越高度不超过 80mm 的障碍物。",
        styles["CNBody"]
    ))
    story.append(Paragraph(
        "机器人搭载了自主研发的 SLAM 2.0 定位导航系统，融合激光雷达、视觉里程计和"
        "惯性测量单元（IMU）数据，实现厘米级定位精度。在无 GPS 信号的室内环境中，"
        "定位误差不超过 ±2cm。同时，机器人配备 5G 通信模块，支持超低延迟的远程操控"
        "和实时视频回传，端到端延迟控制在 50ms 以内。",
        styles["CNBody"]
    ))

    story.append(Paragraph("1.2  应用场景", styles["CNH2"]))
    story.append(Paragraph(
        "XJ-8000 系列机器人已在多个行业领域实现规模化部署。截至 2025 年 11 月，"
        "累计出货量突破 1200 台，覆盖全国 28 个省份。主要应用场景包括：变电站设备"
        "红外测温与可见光巡检、石化厂区气体泄漏检测与管道腐蚀评估、数据中心机房"
        "温湿度监测与服务器状态巡检、矿山巷道结构变形监测与瓦斯浓度检测、以及大型"
        "仓储物流中心的消防设施巡检与货架安全检查。",
        styles["CNBody"]
    ))

    story.append(Paragraph("1.3  核心优势", styles["CNH2"]))

    # 核心优势对比表
    advantage_data = [
        [Paragraph("<b>对比维度</b>", styles["CNTableHeader"]),
         Paragraph("<b>XJ-8000</b>", styles["CNTableHeader"]),
         Paragraph("<b>行业平均水平</b>", styles["CNTableHeader"]),
         Paragraph("<b>领先幅度</b>", styles["CNTableHeader"])],
        [Paragraph("续航时间", styles["CNTableCell"]),
         Paragraph("8-12 小时", styles["CNTableCell"]),
         Paragraph("4-6 小时", styles["CNTableCell"]),
         Paragraph("+100%", styles["CNTableCell"])],
        [Paragraph("定位精度", styles["CNTableCell"]),
         Paragraph("±2cm", styles["CNTableCell"]),
         Paragraph("±5cm", styles["CNTableCell"]),
         Paragraph("+150%", styles["CNTableCell"])],
        [Paragraph("红外测温精度", styles["CNTableCell"]),
         Paragraph("±0.3℃", styles["CNTableCell"]),
         Paragraph("±1.0℃", styles["CNTableCell"]),
         Paragraph("+233%", styles["CNTableCell"])],
        [Paragraph("防护等级", styles["CNTableCell"]),
         Paragraph("IP67", styles["CNTableCell"]),
         Paragraph("IP54", styles["CNTableCell"]),
         Paragraph("3 个等级", styles["CNTableCell"])],
        [Paragraph("MTBF（平均无故障时间）", styles["CNTableCell"]),
         Paragraph("5000 小时", styles["CNTableCell"]),
         Paragraph("2000 小时", styles["CNTableCell"]),
         Paragraph("+150%", styles["CNTableCell"])],
    ]
    adv_table = Table(advantage_data, colWidths=[3.5 * cm, 3.5 * cm, 3.5 * cm, 3.5 * cm])
    adv_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#16213e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), white),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#cccccc")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, HexColor("#f0f4ff")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("BACKGROUND", (3, 1), (3, -1), HexColor("#e8f5e9")),
    ]))
    story.append(adv_table)
    story.append(Paragraph("表 1-1：XJ-8000 核心优势对比", styles["CNNote"]))
    story.append(PageBreak())

    # ============================================================
    # 第2章：技术参数
    # ============================================================
    story.append(Paragraph("2  技术参数", styles["CNH1"]))
    story.append(make_horizontal_line())

    story.append(Paragraph("2.1  硬件规格", styles["CNH2"]))

    # 硬件规格表（含合并单元格）
    hw_data = [
        [Paragraph("<b>类别</b>", styles["CNTableHeader"]),
         Paragraph("<b>参数项</b>", styles["CNTableHeader"]),
         Paragraph("<b>规格值</b>", styles["CNTableHeader"]),
         Paragraph("<b>备注</b>", styles["CNTableHeader"])],
        [Paragraph("尺寸重量", styles["CNTableCell"]),
         Paragraph("外形尺寸", styles["CNTableCell"]),
         Paragraph("680×520×1100mm", styles["CNTableCell"]),
         Paragraph("含天线", styles["CNTableCell"])],
        [Paragraph("尺寸重量", styles["CNTableCell"]),
         Paragraph("整机重量", styles["CNTableCell"]),
         Paragraph("42kg", styles["CNTableCell"]),
         Paragraph("含电池", styles["CNTableCell"])],
        [Paragraph("尺寸重量", styles["CNTableCell"]),
         Paragraph("底盘离地间隙", styles["CNTableCell"]),
         Paragraph("120mm", styles["CNTableCell"]),
         Paragraph("可调 ±20mm", styles["CNTableCell"])],
        [Paragraph("动力系统", styles["CNTableCell"]),
         Paragraph("驱动方式", styles["CNTableCell"]),
         Paragraph("四轮独立驱动", styles["CNTableCell"]),
         Paragraph("轮毂电机", styles["CNTableCell"])],
        [Paragraph("动力系统", styles["CNTableCell"]),
         Paragraph("最大速度", styles["CNTableCell"]),
         Paragraph("1.5m/s", styles["CNTableCell"]),
         Paragraph("可编程限速", styles["CNTableCell"])],
        [Paragraph("动力系统", styles["CNTableCell"]),
         Paragraph("最大爬坡角度", styles["CNTableCell"]),
         Paragraph("30°", styles["CNTableCell"]),
         Paragraph("满载状态", styles["CNTableCell"])],
        [Paragraph("电源系统", styles["CNTableCell"]),
         Paragraph("电池类型", styles["CNTableCell"]),
         Paragraph("磷酸铁锂 48V/60Ah", styles["CNTableCell"]),
         Paragraph("可更换设计", styles["CNTableCell"])],
        [Paragraph("电源系统", styles["CNTableCell"]),
         Paragraph("充电方式", styles["CNTableCell"]),
         Paragraph("无线充电 + 有线快充", styles["CNTableCell"]),
         Paragraph("自动回充", styles["CNTableCell"])],
        [Paragraph("电源系统", styles["CNTableCell"]),
         Paragraph("充电时间", styles["CNTableCell"]),
         Paragraph("无线 4h / 有线 1.5h", styles["CNTableCell"]),
         Paragraph("0-80% SOC", styles["CNTableCell"])],
    ]
    hw_table = Table(hw_data, colWidths=[2.5 * cm, 3.5 * cm, 4.5 * cm, 3.5 * cm])

    # 合并同类单元格
    merge_ranges = [
        ("SPAN", (0, 1), (0, 3)),   # 尺寸重量
        ("SPAN", (0, 4), (0, 6)),   # 动力系统
        ("SPAN", (0, 7), (0, 9)),   # 电源系统
    ]
    base_style = [
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#16213e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), white),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#cccccc")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, HexColor("#f5f7fa")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]
    hw_table.setStyle(TableStyle(base_style + merge_ranges))
    story.append(hw_table)
    story.append(Paragraph("表 2-1：XJ-8000 硬件规格参数", styles["CNNote"]))

    story.append(Paragraph("2.2  传感器配置", styles["CNH2"]))
    story.append(Paragraph(
        "XJ-8000 配备了业界领先的多传感器融合系统，包括 1 台 32 线机械式激光雷达"
        "（探测距离 150m，精度 ±2cm）、2 台固态激光雷达（用于近距离避障）、1 台"
        "高分辨率红外热成像仪（分辨率 640×512，测温范围 -20℃ 至 +550℃，精度 ±0.3℃）、"
        "1 台 4K 可见光云台摄像机（30 倍光学变焦，支持自动聚焦和智能追踪）、4 路"
        "超声波传感器（探测距离 0.02-4m）、1 套六轴 IMU 惯性测量单元（陀螺仪零偏"
        "稳定性 2°/h）以及 1 套多气体检测模块（可同时检测 CH4、H2S、CO、O2、VOC "
        "等 8 种气体，分辨率达到 ppb 级别）。",
        styles["CNBody"]
    ))

    # 传感器配置表
    sensor_data = [
        [Paragraph("<b>序号</b>", styles["CNTableHeader"]),
         Paragraph("<b>传感器类型</b>", styles["CNTableHeader"]),
         Paragraph("<b>型号/规格</b>", styles["CNTableHeader"]),
         Paragraph("<b>数量</b>", styles["CNTableHeader"]),
         Paragraph("<b>核心指标</b>", styles["CNTableHeader"])],
        [Paragraph("1", styles["CNTableCell"]),
         Paragraph("机械式激光雷达", styles["CNTableCell"]),
         Paragraph("RS-Helios-32", styles["CNTableCell"]),
         Paragraph("1", styles["CNTableCell"]),
         Paragraph("32线/150m/±2cm", styles["CNTableCell"])],
        [Paragraph("2", styles["CNTableCell"]),
         Paragraph("固态激光雷达", styles["CNTableCell"]),
         Paragraph("Livox Mid-70", styles["CNTableCell"]),
         Paragraph("2", styles["CNTableCell"]),
         Paragraph("FOV 70°/近距离避障", styles["CNTableCell"])],
        [Paragraph("3", styles["CNTableCell"]),
         Paragraph("红外热成像仪", styles["CNTableCell"]),
         Paragraph("FLIR A700", styles["CNTableCell"]),
         Paragraph("1", styles["CNTableCell"]),
         Paragraph("640×512/±0.3℃", styles["CNTableCell"])],
        [Paragraph("4", styles["CNTableCell"]),
         Paragraph("可见光云台相机", styles["CNTableCell"]),
         Paragraph("Hikvision DS-2DF", styles["CNTableCell"]),
         Paragraph("1", styles["CNTableCell"]),
         Paragraph("4K/30x光学变焦", styles["CNTableCell"])],
        [Paragraph("5", styles["CNTableCell"]),
         Paragraph("超声波传感器", styles["CNTableCell"]),
         Paragraph("HC-SR04 Plus", styles["CNTableCell"]),
         Paragraph("4", styles["CNTableCell"]),
         Paragraph("0.02-4m探测范围", styles["CNTableCell"])],
        [Paragraph("6", styles["CNTableCell"]),
         Paragraph("IMU惯性单元", styles["CNTableCell"]),
         Paragraph("BMI088定制", styles["CNTableCell"]),
         Paragraph("1", styles["CNTableCell"]),
         Paragraph("六轴/2°/h零偏", styles["CNTableCell"])],
        [Paragraph("7", styles["CNTableCell"]),
         Paragraph("多气体检测模块", styles["CNTableCell"]),
         Paragraph("CUBIC-8X", styles["CNTableCell"]),
         Paragraph("1", styles["CNTableCell"]),
         Paragraph("8种气体/ppb级", styles["CNTableCell"])],
        [Paragraph("8", styles["CNTableCell"]),
         Paragraph("温湿度传感器", styles["CNTableCell"]),
         Paragraph("SHT35-DIS", styles["CNTableCell"]),
         Paragraph("2", styles["CNTableCell"]),
         Paragraph("±0.1℃/±1.5%RH", styles["CNTableCell"])],
        [Paragraph("9", styles["CNTableCell"]),
         Paragraph("噪声传感器", styles["CNTableCell"]),
         Paragraph("AWA5636", styles["CNTableCell"]),
         Paragraph("1", styles["CNTableCell"]),
         Paragraph("30-130dB/A计权", styles["CNTableCell"])],
        [Paragraph("10", styles["CNTableCell"]),
         Paragraph("振动传感器", styles["CNTableCell"]),
         Paragraph("ADXL355", styles["CNTableCell"]),
         Paragraph("4", styles["CNTableCell"]),
         Paragraph("三轴/0.01g分辨率", styles["CNTableCell"])],
    ]
    sensor_table = Table(sensor_data, colWidths=[1.2 * cm, 3.2 * cm, 3.2 * cm, 1.2 * cm, 5.2 * cm])
    sensor_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#16213e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), white),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#cccccc")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, HexColor("#f5f7fa")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 0), (0, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(sensor_table)
    story.append(Paragraph("表 2-2：传感器配置清单", styles["CNNote"]))

    story.append(Paragraph("2.3  通信与接口", styles["CNH2"]))
    story.append(Paragraph(
        "机器人通信系统支持 5G SA/NSA 双模、Wi-Fi 6（802.11ax）双频、LoRa 远距离"
        "低功耗通信（最大传输距离 10km 视距）以及 1000Mbps 有线以太网接口。外部接口"
        "包括 2 路 USB 3.0 Type-A、1 路 USB-C（支持 DP Alt Mode 视频输出）、1 路 "
        "HDMI 2.0、1 路 RS-485 工业总线接口和 1 路 CAN FD 总线接口。软件层面支持 "
        "MQTT 3.1.1/5.0、OPC UA、Modbus TCP/RTU 等工业物联网协议，可与主流 SCADA "
        "系统和 MES 系统无缝对接。",
        styles["CNBody"]
    ))
    story.append(PageBreak())

    # ============================================================
    # 第3章：性能测试数据
    # ============================================================
    story.append(Paragraph("3  性能测试数据", styles["CNH1"]))
    story.append(make_horizontal_line())

    story.append(Paragraph("3.1  续航与充电性能", styles["CNH2"]))
    story.append(Paragraph(
        "续航测试在标准测试环境下进行：环境温度 25℃，平坦路面，匀速 1.0m/s 行驶，"
        "所有传感器全开，5G 通信保持连接。测试结果表明，XJ-8000 在满载传感器工作"
        "状态下，单次充电可持续运行 8.5 至 12 小时（视行驶速度与传感器工作负载而定），"
        "最大行驶里程达到 28km。在 -20℃ 低温环境下，续航时间约为常温的 72%，仍可"
        "满足一个完整班次（8 小时）的巡检需求。",
        styles["CNBody"]
    ))

    # 续航测试数据表
    battery_data = [
        [Paragraph("<b>测试工况</b>", styles["CNTableHeader"]),
         Paragraph("<b>环境温度</b>", styles["CNTableHeader"]),
         Paragraph("<b>平均功耗</b>", styles["CNTableHeader"]),
         Paragraph("<b>续航时间</b>", styles["CNTableHeader"]),
         Paragraph("<b>行驶里程</b>", styles["CNTableHeader"])],
        [Paragraph("标准巡检", styles["CNTableCell"]), Paragraph("25℃", styles["CNTableCell"]),
         Paragraph("380W", styles["CNTableCell"]), Paragraph("10.2h", styles["CNTableCell"]),
         Paragraph("26.5km", styles["CNTableCell"])],
        [Paragraph("高速巡检", styles["CNTableCell"]), Paragraph("25℃", styles["CNTableCell"]),
         Paragraph("520W", styles["CNTableCell"]), Paragraph("8.5h", styles["CNTableCell"]),
         Paragraph("28.0km", styles["CNTableCell"])],
        [Paragraph("低温巡检", styles["CNTableCell"]), Paragraph("-20℃", styles["CNTableCell"]),
         Paragraph("450W", styles["CNTableCell"]), Paragraph("7.3h", styles["CNTableCell"]),
         Paragraph("18.2km", styles["CNTableCell"])],
        [Paragraph("高温巡检", styles["CNTableCell"]), Paragraph("55℃", styles["CNTableCell"]),
         Paragraph("420W", styles["CNTableCell"]), Paragraph("8.9h", styles["CNTableCell"]),
         Paragraph("22.8km", styles["CNTableCell"])],
        [Paragraph("待机监测", styles["CNTableCell"]), Paragraph("25℃", styles["CNTableCell"]),
         Paragraph("120W", styles["CNTableCell"]), Paragraph("24.0h", styles["CNTableCell"]),
         Paragraph("—", styles["CNTableCell"])],
    ]
    battery_table = Table(battery_data, colWidths=[2.5 * cm, 2.5 * cm, 2.5 * cm, 2.5 * cm, 2.5 * cm])
    battery_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#16213e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), white),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#cccccc")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, HexColor("#f5f7fa")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(battery_table)
    story.append(Paragraph("表 3-1：续航与充电性能测试数据", styles["CNNote"]))

    story.append(Paragraph("3.2  检测精度", styles["CNH2"]))
    story.append(Paragraph(
        "检测精度是巡检机器人的核心性能指标。XJ-8000 在标准测试集上的表现如下："
        "仪表读数识别准确率达到 99.2%（测试样本 5000 张各类工业仪表图片），设备"
        "外观缺陷检测召回率 97.8%、精确率 96.5%（涵盖锈蚀、裂纹、变形、松动等 12 类"
        "常见缺陷），红外热斑检测灵敏度达到 0.05℃（NETD），气体泄漏检测最低检出限"
        "为 1ppm（甲烷）。所有检测算法均基于自研的 DeepInspect 3.0 深度学习框架，"
        "模型推理在机器人搭载的 NVIDIA Jetson Orin NX 边缘计算模块上完成，单帧推理"
        "时间不超过 35ms。",
        styles["CNBody"]
    ))

    # 检测精度表
    accuracy_data = [
        [Paragraph("<b>检测项目</b>", styles["CNTableHeader"]),
         Paragraph("<b>指标</b>", styles["CNTableHeader"]),
         Paragraph("<b>数值</b>", styles["CNTableHeader"]),
         Paragraph("<b>测试集规模</b>", styles["CNTableHeader"])],
        [Paragraph("仪表读数识别", styles["CNTableLeft"]), Paragraph("准确率", styles["CNTableCell"]),
         Paragraph("99.2%", styles["CNTableCell"]), Paragraph("5000张", styles["CNTableCell"])],
        [Paragraph("设备缺陷检测", styles["CNTableLeft"]), Paragraph("召回率", styles["CNTableCell"]),
         Paragraph("97.8%", styles["CNTableCell"]), Paragraph("12000张", styles["CNTableCell"])],
        [Paragraph("设备缺陷检测", styles["CNTableLeft"]), Paragraph("精确率", styles["CNTableCell"]),
         Paragraph("96.5%", styles["CNTableCell"]), Paragraph("12000张", styles["CNTableCell"])],
        [Paragraph("红外热斑检测", styles["CNTableLeft"]), Paragraph("NETD", styles["CNTableCell"]),
         Paragraph("0.05℃", styles["CNTableCell"]), Paragraph("800组", styles["CNTableCell"])],
        [Paragraph("气体泄漏检测", styles["CNTableLeft"]), Paragraph("最低检出限", styles["CNTableCell"]),
         Paragraph("1ppm", styles["CNTableCell"]), Paragraph("500组", styles["CNTableCell"])],
        [Paragraph("声纹异常检测", styles["CNTableLeft"]), Paragraph("F1-Score", styles["CNTableCell"]),
         Paragraph("94.3%", styles["CNTableCell"]), Paragraph("3000段", styles["CNTableCell"])],
    ]
    accuracy_table = Table(accuracy_data, colWidths=[3 * cm, 2.5 * cm, 2.5 * cm, 2.5 * cm])
    accuracy_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#16213e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), white),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#cccccc")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, HexColor("#f5f7fa")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("SPAN", (0, 2), (0, 3)),
    ]))
    story.append(accuracy_table)
    story.append(Paragraph("表 3-2：检测精度指标汇总", styles["CNNote"]))

    story.append(Paragraph("3.3  环境适应性", styles["CNH2"]))

    # 环境适应性测试表
    env_data = [
        [Paragraph("<b>测试项目</b>", styles["CNTableHeader"]),
         Paragraph("<b>测试标准</b>", styles["CNTableHeader"]),
         Paragraph("<b>测试条件</b>", styles["CNTableHeader"]),
         Paragraph("<b>结果</b>", styles["CNTableHeader"])],
        [Paragraph("高温运行", styles["CNTableLeft"]), Paragraph("GB/T 2423.2", styles["CNTableCell"]),
         Paragraph("+65℃/16h", styles["CNTableCell"]), Paragraph("通过", styles["CNTableCell"])],
        [Paragraph("低温运行", styles["CNTableLeft"]), Paragraph("GB/T 2423.1", styles["CNTableCell"]),
         Paragraph("-30℃/16h", styles["CNTableCell"]), Paragraph("通过", styles["CNTableCell"])],
        [Paragraph("湿热循环", styles["CNTableLeft"]), Paragraph("GB/T 2423.3", styles["CNTableCell"]),
         Paragraph("95%RH/55℃/48h", styles["CNTableCell"]), Paragraph("通过", styles["CNTableCell"])],
        [Paragraph("盐雾腐蚀", styles["CNTableLeft"]), Paragraph("GB/T 2423.17", styles["CNTableCell"]),
         Paragraph("5%NaCl/96h", styles["CNTableCell"]), Paragraph("通过", styles["CNTableCell"])],
        [Paragraph("振动测试", styles["CNTableLeft"]), Paragraph("GB/T 2423.10", styles["CNTableCell"]),
         Paragraph("10-500Hz/5g", styles["CNTableCell"]), Paragraph("通过", styles["CNTableCell"])],
        [Paragraph("冲击测试", styles["CNTableLeft"]), Paragraph("GB/T 2423.5", styles["CNTableCell"]),
         Paragraph("30g/11ms", styles["CNTableCell"]), Paragraph("通过", styles["CNTableCell"])],
        [Paragraph("IP67防水", styles["CNTableLeft"]), Paragraph("GB/T 4208", styles["CNTableCell"]),
         Paragraph("1m水深/30min", styles["CNTableCell"]), Paragraph("通过", styles["CNTableCell"])],
        [Paragraph("IP67防尘", styles["CNTableLeft"]), Paragraph("GB/T 4208", styles["CNTableCell"]),
         Paragraph("滑石粉/8h", styles["CNTableCell"]), Paragraph("通过", styles["CNTableCell"])],
    ]
    env_table = Table(env_data, colWidths=[2.5 * cm, 3 * cm, 3.5 * cm, 2 * cm])
    env_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#16213e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), white),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#cccccc")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, HexColor("#f5f7fa")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("BACKGROUND", (3, 1), (3, -1), HexColor("#e8f5e9")),
    ]))
    story.append(env_table)
    story.append(Paragraph("表 3-3：环境适应性测试结果", styles["CNNote"]))
    story.append(PageBreak())

    # ============================================================
    # 第4章：系统架构（含图示）
    # ============================================================
    story.append(Paragraph("4  系统架构", styles["CNH1"]))
    story.append(make_horizontal_line())

    story.append(Paragraph(
        "XJ-8000 系统采用分层架构设计，自底向上分为四层：感知层、计算层、通信层和应用层。"
        "各层之间通过标准化接口进行数据交互，实现了高内聚、低耦合的架构目标。",
        styles["CNBody"]
    ))

    # 系统架构图（使用 Drawing 绘制）
    d_arch = Drawing(480, 280)
    # 背景
    d_arch.add(Rect(0, 0, 480, 280, fillColor=HexColor("#fafbfc"), strokeColor=HexColor("#dddddd"), strokeWidth=0.5))

    # 应用层
    d_arch.add(Rect(40, 220, 400, 45, fillColor=HexColor("#e3f2fd"), strokeColor=HexColor("#1565c0"), strokeWidth=1.5))
    d_arch.add(String(240, 237, "应用层：巡检任务调度 / 数据分析报表 / 告警管理 / 远程操控", fontName="SimHei", fontSize=9, fillColor=HexColor("#1565c0"), textAnchor="middle"))

    # 箭头
    d_arch.add(Line(240, 220, 240, 205, strokeColor=HexColor("#666666"), strokeWidth=1))
    d_arch.add(Polygon([235, 208, 240, 200, 245, 208], fillColor=HexColor("#666666"), strokeColor=HexColor("#666666")))

    # 通信层
    d_arch.add(Rect(40, 155, 400, 45, fillColor=HexColor("#fff3e0"), strokeColor=HexColor("#e65100"), strokeWidth=1.5))
    d_arch.add(String(240, 172, "通信层：5G / Wi-Fi 6 / LoRa / MQTT / OPC UA / Modbus", fontName="SimHei", fontSize=9, fillColor=HexColor("#e65100"), textAnchor="middle"))

    # 箭头
    d_arch.add(Line(240, 155, 240, 140, strokeColor=HexColor("#666666"), strokeWidth=1))
    d_arch.add(Polygon([235, 143, 240, 135, 245, 143], fillColor=HexColor("#666666"), strokeColor=HexColor("#666666")))

    # 计算层
    d_arch.add(Rect(40, 90, 400, 45, fillColor=HexColor("#e8f5e9"), strokeColor=HexColor("#2e7d32"), strokeWidth=1.5))
    d_arch.add(String(240, 107, "计算层：NVIDIA Jetson Orin NX / SLAM 2.0 / DeepInspect 3.0 / 边缘推理", fontName="SimHei", fontSize=9, fillColor=HexColor("#2e7d32"), textAnchor="middle"))

    # 箭头
    d_arch.add(Line(240, 90, 240, 75, strokeColor=HexColor("#666666"), strokeWidth=1))
    d_arch.add(Polygon([235, 78, 240, 70, 245, 78], fillColor=HexColor("#666666"), strokeColor=HexColor("#666666")))

    # 感知层
    d_arch.add(Rect(40, 25, 400, 45, fillColor=HexColor("#f3e5f5"), strokeColor=HexColor("#7b1fa2"), strokeWidth=1.5))
    d_arch.add(String(240, 42, "感知层：激光雷达 / 红外热像 / 可见光相机 / 气体检测 / IMU / 超声波", fontName="SimHei", fontSize=9, fillColor=HexColor("#7b1fa2"), textAnchor="middle"))

    story.append(d_arch)
    story.append(Paragraph("图 4-1：XJ-8000 系统分层架构图", styles["CNNote"]))

    story.append(Paragraph(
        "感知层负责采集环境数据，包括激光点云、红外热图、可见光图像、气体浓度、"
        "温湿度、噪声和振动等多模态信息。计算层基于 NVIDIA Jetson Orin NX 平台（算力"
        "100 TOPS），运行 SLAM 2.0 定位导航算法和 DeepInspect 3.0 智能检测算法，"
        "实现实时环境感知与设备状态分析。通信层提供多通道冗余通信能力，确保在复杂"
        "电磁环境中数据传输的可靠性。应用层通过 Web 管理平台和移动 APP 提供任务配置、"
        "实时监控、历史回溯和报表生成等功能。",
        styles["CNBody"]
    ))

    # 数据流示意图
    d_flow = Drawing(480, 100)
    d_flow.add(Rect(0, 0, 480, 100, fillColor=HexColor("#fafbfc"), strokeColor=HexColor("#dddddd"), strokeWidth=0.5))

    boxes = [
        (20, 35, 80, 30, "传感器采集", "#e3f2fd", "#1565c0"),
        (130, 35, 80, 30, "数据预处理", "#fff3e0", "#e65100"),
        (240, 35, 80, 30, "AI推理分析", "#e8f5e9", "#2e7d32"),
        (350, 35, 80, 30, "结果上报", "#f3e5f5", "#7b1fa2"),
    ]
    for x, y, w, h, label, fill, stroke in boxes:
        d_flow.add(Rect(x, y, w, h, fillColor=HexColor(fill), strokeColor=HexColor(stroke), strokeWidth=1))
        d_flow.add(String(x + w / 2, y + h / 2 - 4, label, fontName="SimHei", fontSize=8, fillColor=HexColor(stroke), textAnchor="middle"))

    # 箭头
    for sx in [100, 210, 320]:
        d_flow.add(Line(sx, 50, sx + 25, 50, strokeColor=HexColor("#999999"), strokeWidth=1.5))
        d_flow.add(Polygon([sx + 22, 46, sx + 30, 50, sx + 22, 54], fillColor=HexColor("#999999"), strokeColor=HexColor("#999999")))

    story.append(d_flow)
    story.append(Paragraph("图 4-2：数据处理流水线", styles["CNNote"]))
    story.append(PageBreak())

    # ============================================================
    # 第5章：部署方案对比
    # ============================================================
    story.append(Paragraph("5  部署方案对比", styles["CNH1"]))
    story.append(make_horizontal_line())

    story.append(Paragraph(
        "根据客户现场条件和巡检需求，XJ-8000 提供三种标准部署方案：单机独立部署、"
        "多机协同部署和云端融合部署。以下从多个维度对三种方案进行详细对比。",
        styles["CNBody"]
    ))

    deploy_data = [
        [Paragraph("<b>对比维度</b>", styles["CNTableHeader"]),
         Paragraph("<b>单机独立部署</b>", styles["CNTableHeader"]),
         Paragraph("<b>多机协同部署</b>", styles["CNTableHeader"]),
         Paragraph("<b>云端融合部署</b>", styles["CNTableHeader"])],
        [Paragraph("适用场景", styles["CNTableLeft"]),
         Paragraph("小型站点\n面积<5000m²", styles["CNTableCell"]),
         Paragraph("中型场站\n5000-50000m²", styles["CNTableCell"]),
         Paragraph("大型园区\n>50000m²", styles["CNTableCell"])],
        [Paragraph("机器人数量", styles["CNTableLeft"]),
         Paragraph("1台", styles["CNTableCell"]),
         Paragraph("2-8台", styles["CNTableCell"]),
         Paragraph("8台以上", styles["CNTableCell"])],
        [Paragraph("部署周期", styles["CNTableLeft"]),
         Paragraph("1-2天", styles["CNTableCell"]),
         Paragraph("3-7天", styles["CNTableCell"]),
         Paragraph("7-15天", styles["CNTableCell"])],
        [Paragraph("网络要求", styles["CNTableLeft"]),
         Paragraph("本地Wi-Fi即可", styles["CNTableCell"]),
         Paragraph("5G专网推荐", styles["CNTableCell"]),
         Paragraph("5G+边缘云", styles["CNTableCell"])],
        [Paragraph("数据存储", styles["CNTableLeft"]),
         Paragraph("本地SSD\n512GB", styles["CNTableCell"]),
         Paragraph("本地+NAS\n2TB", styles["CNTableCell"]),
         Paragraph("云端存储\n按需扩容", styles["CNTableCell"])],
        [Paragraph("AI算力", styles["CNTableLeft"]),
         Paragraph("边缘端\n100TOPS", styles["CNTableCell"]),
         Paragraph("边缘+协同\n200TOPS", styles["CNTableCell"]),
         Paragraph("云边协同\n500+TOPS", styles["CNTableCell"])],
        [Paragraph("单台成本", styles["CNTableLeft"]),
         Paragraph("28万元", styles["CNTableCell"]),
         Paragraph("25万元/台\n（批量优惠）", styles["CNTableCell"]),
         Paragraph("22万元/台\n（含云服务费）", styles["CNTableCell"])],
        [Paragraph("年度维护费", styles["CNTableLeft"]),
         Paragraph("1.5万元", styles["CNTableCell"]),
         Paragraph("1.2万元/台", styles["CNTableCell"]),
         Paragraph("0.8万元/台\n（远程运维）", styles["CNTableCell"])],
    ]
    deploy_table = Table(deploy_data, colWidths=[2.8 * cm, 3.2 * cm, 3.2 * cm, 3.2 * cm])
    deploy_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#16213e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), white),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#cccccc")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, HexColor("#f5f7fa")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("BACKGROUND", (0, 1), (0, -1), HexColor("#f0f4ff")),
    ]))
    story.append(deploy_table)
    story.append(Paragraph("表 5-1：三种部署方案详细对比", styles["CNNote"]))

    story.append(Paragraph(
        "对于初次使用巡检机器人的客户，推荐从单机独立部署方案开始，在熟悉系统功能和"
        "操作流程后，可根据实际需求平滑升级至多机协同或云端融合方案。所有方案均支持"
        "数据无缝迁移，升级过程中无需中断现有巡检任务。",
        styles["CNBody"]
    ))
    story.append(PageBreak())

    # ============================================================
    # 第6章：维护保养指南
    # ============================================================
    story.append(Paragraph("6  维护保养指南", styles["CNH1"]))
    story.append(make_horizontal_line())

    story.append(Paragraph(
        "规范的维护保养是确保机器人长期稳定运行的关键。本章详细说明了 XJ-8000 系列"
        "机器人的日常维护、定期保养和易损件更换的操作规程与周期要求。",
        styles["CNBody"]
    ))

    # 维护周期表
    maint_data = [
        [Paragraph("<b>维护项目</b>", styles["CNTableHeader"]),
         Paragraph("<b>周期</b>", styles["CNTableHeader"]),
         Paragraph("<b>操作内容</b>", styles["CNTableHeader"]),
         Paragraph("<b>责任人</b>", styles["CNTableHeader"]),
         Paragraph("<b>耗时</b>", styles["CNTableHeader"])],
        [Paragraph("外观清洁", styles["CNTableLeft"]),
         Paragraph("每日", styles["CNTableCell"]),
         Paragraph("擦拭外壳、清洁传感器窗口、检查天线紧固", styles["CNTableLeft"]),
         Paragraph("操作员", styles["CNTableCell"]),
         Paragraph("10min", styles["CNTableCell"])],
        [Paragraph("轮胎检查", styles["CNTableLeft"]),
         Paragraph("每周", styles["CNTableCell"]),
         Paragraph("检查胎压、胎面磨损、轮毂螺栓扭矩（45N·m）", styles["CNTableLeft"]),
         Paragraph("操作员", styles["CNTableCell"]),
         Paragraph("15min", styles["CNTableCell"])],
        [Paragraph("传感器校准", styles["CNTableLeft"]),
         Paragraph("每月", styles["CNTableCell"]),
         Paragraph("激光雷达水平校准、红外测温黑体校准、气体传感器零点校准", styles["CNTableLeft"]),
         Paragraph("技术员", styles["CNTableCell"]),
         Paragraph("45min", styles["CNTableCell"])],
        [Paragraph("电池检测", styles["CNTableLeft"]),
         Paragraph("每季度", styles["CNTableCell"]),
         Paragraph("内阻测试、容量衰减评估、BMS固件升级、均衡充电", styles["CNTableLeft"]),
         Paragraph("技术员", styles["CNTableCell"]),
         Paragraph("60min", styles["CNTableCell"])],
        [Paragraph("全面检修", styles["CNTableLeft"]),
         Paragraph("每半年", styles["CNTableCell"]),
         Paragraph("全系统诊断、电机编码器校准、通信模块测试、防水密封更换", styles["CNTableLeft"]),
         Paragraph("工程师", styles["CNTableCell"]),
         Paragraph("4h", styles["CNTableCell"])],
        [Paragraph("大修保养", styles["CNTableLeft"]),
         Paragraph("每年", styles["CNTableCell"]),
         Paragraph("减速器换油、轴承更换、线束检查、结构件探伤、系统软件大版本升级", styles["CNTableLeft"]),
         Paragraph("厂商工程师", styles["CNTableCell"]),
         Paragraph("8h", styles["CNTableCell"])],
    ]
    maint_table = Table(maint_data, colWidths=[2.5 * cm, 2 * cm, 5.5 * cm, 2.2 * cm, 1.8 * cm])
    maint_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#16213e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), white),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#cccccc")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, HexColor("#f5f7fa")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(maint_table)
    story.append(Paragraph("表 6-1：定期维护保养计划表", styles["CNNote"]))

    story.append(Paragraph(
        "所有维护操作均需在《设备维护记录表》中登记，记录维护日期、操作内容、"
        "更换的零部件型号与批次号、操作人员签名以及下次维护到期日期。对于涉及"
        "安全功能的维护项目（如电池检测、防水密封更换），必须由经过厂商认证的"
        "技术人员执行，并在维护完成后进行功能验证测试。",
        styles["CNBody"]
    ))

    story.append(Paragraph("6.1  易损件更换周期", styles["CNH2"]))

    parts_data = [
        [Paragraph("<b>零部件名称</b>", styles["CNTableHeader"]),
         Paragraph("<b>规格型号</b>", styles["CNTableHeader"]),
         Paragraph("<b>设计寿命</b>", styles["CNTableHeader"]),
         Paragraph("<b>建议更换周期</b>", styles["CNTableHeader"]),
         Paragraph("<b>单价（元）</b>", styles["CNTableHeader"])],
        [Paragraph("驱动轮胎", styles["CNTableLeft"]), Paragraph("XJ-TIRE-12", styles["CNTableCell"]),
         Paragraph("2000km", styles["CNTableCell"]), Paragraph("1500km 或 12个月", styles["CNTableCell"]),
         Paragraph("680", styles["CNTableCell"])],
        [Paragraph("激光雷达窗口片", styles["CNTableLeft"]), Paragraph("XJ-WIN-32", styles["CNTableCell"]),
         Paragraph("2年", styles["CNTableCell"]), Paragraph("18个月", styles["CNTableCell"]),
         Paragraph("420", styles["CNTableCell"])],
        [Paragraph("空气滤清器", styles["CNTableLeft"]), Paragraph("XJ-FIL-H13", styles["CNTableCell"]),
         Paragraph("6个月", styles["CNTableCell"]), Paragraph("3个月（恶劣环境）", styles["CNTableCell"]),
         Paragraph("180", styles["CNTableCell"])],
        [Paragraph("电池模组", styles["CNTableLeft"]), Paragraph("XJ-BAT-48V60", styles["CNTableCell"]),
         Paragraph("1500次循环", styles["CNTableCell"]), Paragraph("1200次循环 或 3年", styles["CNTableCell"]),
         Paragraph("8500", styles["CNTableCell"])],
        [Paragraph("减速器润滑油", styles["CNTableLeft"]), Paragraph("XJ-OIL-SYN", styles["CNTableCell"]),
         Paragraph("1年", styles["CNTableCell"]), Paragraph("每年更换", styles["CNTableCell"]),
         Paragraph("320", styles["CNTableCell"])],
        [Paragraph("防水密封圈套件", styles["CNTableLeft"]), Paragraph("XJ-SEAL-KIT", styles["CNTableCell"]),
         Paragraph("2年", styles["CNTableCell"]), Paragraph("每年检查/2年更换", styles["CNTableCell"]),
         Paragraph("560", styles["CNTableCell"])],
    ]
    parts_table = Table(parts_data, colWidths=[3 * cm, 3 * cm, 2.5 * cm, 3.5 * cm, 2 * cm])
    parts_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#16213e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), white),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#cccccc")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, HexColor("#f5f7fa")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(parts_table)
    story.append(Paragraph("表 6-2：易损件更换周期与参考价格", styles["CNNote"]))
    story.append(PageBreak())

    # ============================================================
    # 第7章：故障排查
    # ============================================================
    story.append(Paragraph("7  故障排查", styles["CNH1"]))
    story.append(make_horizontal_line())

    story.append(Paragraph(
        "本章列出了 XJ-8000 系列机器人在实际运行中可能遇到的常见故障现象、"
        "可能原因以及对应的排查步骤和解决方案。操作人员可根据故障代码快速定位问题。",
        styles["CNBody"]
    ))

    fault_data = [
        [Paragraph("<b>故障代码</b>", styles["CNTableHeader"]),
         Paragraph("<b>故障现象</b>", styles["CNTableHeader"]),
         Paragraph("<b>可能原因</b>", styles["CNTableHeader"]),
         Paragraph("<b>排查步骤</b>", styles["CNTableHeader"]),
         Paragraph("<b>解决方案</b>", styles["CNTableHeader"])],
        [Paragraph("E001", styles["CNTableCell"]),
         Paragraph("机器人无法启动", styles["CNTableLeft"]),
         Paragraph("电池亏电/电源模块故障/急停按钮未释放", styles["CNTableLeft"]),
         Paragraph("1.检查急停按钮状态\n2.测量电池电压\n3.检查电源模块指示灯", styles["CNTableLeft"]),
         Paragraph("释放急停按钮；电池电压<42V时充电；更换电源模块", styles["CNTableLeft"])],
        [Paragraph("E015", styles["CNTableCell"]),
         Paragraph("定位漂移/导航异常", styles["CNTableLeft"]),
         Paragraph("激光雷达脏污/IMU零偏超限/环境特征不足", styles["CNTableLeft"]),
         Paragraph("1.清洁激光雷达窗口\n2.执行IMU静态校准\n3.检查SLAM地图质量", styles["CNTableLeft"]),
         Paragraph("清洁传感器；重新建图；在特征稀疏区域增加反光标记", styles["CNTableLeft"])],
        [Paragraph("E023", styles["CNTableCell"]),
         Paragraph("红外测温偏差过大", styles["CNTableLeft"]),
         Paragraph("镜头脏污/环境温度补偿失效/黑体校准过期", styles["CNTableLeft"]),
         Paragraph("1.清洁红外镜头\n2.检查环境温度传感器\n3.查看上次校准日期", styles["CNTableLeft"]),
         Paragraph("清洁镜头；使用标准黑体重新校准；更新温度补偿参数", styles["CNTableLeft"])],
        [Paragraph("E042", styles["CNTableCell"]),
         Paragraph("通信中断/延迟过高", styles["CNTableLeft"]),
         Paragraph("信号遮挡/5G基站切换/天线接口松动", styles["CNTableLeft"]),
         Paragraph("1.检查天线SMA接口\n2.查看信号强度RSRP值\n3.测试ping延迟", styles["CNTableLeft"]),
         Paragraph("紧固天线接口；调整机器人位置避开遮挡；切换至Wi-Fi备用链路", styles["CNTableLeft"])],
        [Paragraph("E058", styles["CNTableCell"]),
         Paragraph("电池续航骤降", styles["CNTableLeft"]),
         Paragraph("电池老化/低温环境影响/传感器异常高功耗", styles["CNTableLeft"]),
         Paragraph("1.查看电池SOH健康度\n2.检查各传感器功耗\n3.导出电池循环记录", styles["CNTableLeft"]),
         Paragraph("SOH<80%时更换电池；低温环境启用电池加热；排查异常功耗传感器", styles["CNTableLeft"])],
    ]
    fault_table = Table(fault_data, colWidths=[1.5 * cm, 2.5 * cm, 3 * cm, 3.5 * cm, 3.5 * cm])
    fault_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#16213e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), white),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#cccccc")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, HexColor("#f5f7fa")]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(fault_table)
    story.append(Paragraph("表 7-1：常见故障代码与排查指南", styles["CNNote"]))

    story.append(Paragraph(
        "当机器人出现上述故障代码时，系统会自动记录故障发生时刻的完整日志（包括传感器"
        "数据快照、系统资源占用情况和通信链路状态），便于技术人员进行远程诊断。对于"
        "E001、E042 等影响基本功能的故障，机器人会自动进入安全停止状态，并通过 5G "
        "网络向管理平台发送紧急告警通知。",
        styles["CNBody"]
    ))

    # 文档结尾
    story.append(Spacer(1, 1 * cm))
    story.append(make_horizontal_line())
    story.append(Paragraph(
        "本文档由技术研发部编制并维护。如有疑问，请联系技术支持热线：400-888-XXXX，"
        "或发送邮件至 support@xj-robot.com。文档版本 V3.2.1，最后更新日期：2025年12月15日。",
        styles["CNNote"]
    ))

    # 构建 PDF
    doc.build(story)
    print(f"复杂 PDF 已生成：{output_path}")
    return output_path


if __name__ == "__main__":
    build_document()
