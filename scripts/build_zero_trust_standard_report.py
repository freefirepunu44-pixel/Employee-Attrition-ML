import os
from pathlib import Path
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
ASSETS_DIR = REPORTS_DIR / "assets_zt"
EDA_DIR = ROOT / "outputs" / "plots" / "eda"
PLOTS_DIR = ROOT / "outputs" / "plots"

VTU_LOGO = ASSETS_DIR / "image1.jpeg"
EWIT_LOGO = ASSETS_DIR / "image2.png"

MAROON = RGBColor(146, 54, 52)
NAVY = RGBColor(31, 58, 95)
DARK = RGBColor(15, 23, 42)
GRAY = RGBColor(100, 116, 139)


def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    tcPr.append(parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>'))


def style_table(table, col_widths=None, header_bg="1F3A5F"):
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, row in enumerate(table.rows):
        trPr = row._tr.get_or_add_trPr()
        trPr.append(parse_xml(f'<w:cantSplit {nsdecls("w")}/>'))
        if i == 0:
            trPr.append(parse_xml(f'<w:tblHeader {nsdecls("w")}/>'))
            for cell in row.cells:
                set_cell_background(cell, header_bg)
                for p in cell.paragraphs:
                    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
                    for run in p.runs:
                        run.font.name = "Times New Roman"
                        run.font.bold = True
                        run.font.color.rgb = RGBColor(255, 255, 255)
                        run.font.size = Pt(10)
        else:
            bg = "F8FAFC" if i % 2 == 1 else "FFFFFF"
            for cell in row.cells:
                set_cell_background(cell, bg)
                for p in cell.paragraphs:
                    for run in p.runs:
                        run.font.name = "Times New Roman"
                        run.font.size = Pt(9.5)
                        run.font.color.rgb = DARK
    if col_widths:
        for row in table.rows:
            for j, w in enumerate(col_widths):
                if j < len(row.cells):
                    row.cells[j].width = Inches(w)


def add_figure_with_caption(doc, img_path, caption, width=Inches(6.0)):
    if Path(img_path).exists():
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img.paragraph_format.space_before = Pt(8)
        p_img.paragraph_format.space_after = Pt(4)
        run = p_img.add_run()
        run.add_picture(str(img_path), width=width)
        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap.paragraph_format.space_after = Pt(12)
        cap_run = p_cap.add_run(caption)
        cap_run.font.name = "Times New Roman"
        cap_run.font.italic = True
        cap_run.font.size = Pt(9.5)
        cap_run.font.color.rgb = GRAY


def build_report():
    print("Building Employee Attrition VTU Project Report using Zero Trust Report Standard...")
    doc = Document()

    # Section 0 Setup (Page margins per VTU standard: Left 1.25", Top 1.0", Right 1.0", Bottom 1.0")
    for sec in doc.sections:
        sec.top_margin = Inches(1.0)
        sec.bottom_margin = Inches(1.0)
        sec.left_margin = Inches(1.25)
        sec.right_margin = Inches(1.0)
        sec.page_width = Inches(8.27)
        sec.page_height = Inches(11.69)

    # Style configuration
    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(12)
    normal.paragraph_format.line_spacing = 1.5
    normal.paragraph_format.space_after = Pt(6)

    # -------------------------------------------------------------
    # 1. COVER PAGE (Exact VTU Cover Format from Zero Trust Report)
    # -------------------------------------------------------------
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(2)
    r = p.add_run("VISVESVARAYA TECHNOLOGICAL UNIVERSITY")
    r.font.name = "Times New Roman"
    r.font.size = Pt(15.5)
    r.font.bold = True
    r.font.color.rgb = DARK

    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_sub.paragraph_format.space_after = Pt(6)
    r_sub = p_sub.add_run("Jnana Sangama, Belgaum - 590014, Karnataka State, India")
    r_sub.font.name = "Times New Roman"
    r_sub.font.size = Pt(10)

    # VTU Logo
    if VTU_LOGO.exists():
        p_logo = doc.add_paragraph()
        p_logo.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_logo.paragraph_format.space_after = Pt(6)
        p_logo.add_run().add_picture(str(VTU_LOGO), width=Inches(1.15))

    p_proj = doc.add_paragraph()
    p_proj.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_proj.paragraph_format.space_after = Pt(4)
    r_rep = p_proj.add_run("A PROJECT REPORT ON")
    r_rep.font.name = "Times New Roman"
    r_rep.font.size = Pt(12)

    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_after = Pt(6)
    r_title = p_title.add_run("EMPLOYEE ATTRITION PREDICTION USING MACHINE LEARNING WITH EXPLAINABLE AI (XAI)")
    r_title.font.name = "Times New Roman"
    r_title.font.size = Pt(15.5)
    r_title.font.bold = True
    r_title.font.color.rgb = DARK

    p_deg = doc.add_paragraph()
    p_deg.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_deg.paragraph_format.space_after = Pt(2)
    r_deg = p_deg.add_run("BACHELOR OF ENGINEERING\nIN\n")
    r_deg.font.name = "Times New Roman"
    r_deg.font.size = Pt(13)
    r_deg.font.bold = True
    r_deg.font.color.rgb = MAROON

    r_dept = p_deg.add_run("CSE (IoT & Cybersecurity including Blockchain)")
    r_dept.font.name = "Times New Roman"
    r_dept.font.size = Pt(13)
    r_dept.font.bold = True
    r_dept.font.color.rgb = DARK

    # Students block
    p_subm = doc.add_paragraph()
    p_subm.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_subm.paragraph_format.space_before = Pt(8)
    p_subm.paragraph_format.space_after = Pt(2)
    r_subm = p_subm.add_run("Submitted by: -")
    r_subm.font.name = "Times New Roman"
    r_subm.font.size = Pt(11.5)
    r_subm.font.bold = True

    stud_table = doc.add_table(rows=4, cols=2)
    stud_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    students = [
        ("ANUSHA H G", "1EW23IC006"),
        ("CHARAN M S", "1EW23IC012"),
        ("NITHIN GOWDA R", "1EW23IC031"),
        ("PUNARVA D N", "1EW23IC037"),
    ]
    for idx, (sname, usn) in enumerate(students):
        row = stud_table.rows[idx]
        p_n = row.cells[0].paragraphs[0]
        p_n.alignment = WD_ALIGN_PARAGRAPH.LEFT
        rn = p_n.add_run(f"{idx+1}.  {sname}")
        rn.font.name = "Times New Roman"
        rn.font.size = Pt(11)
        rn.font.bold = True

        p_u = row.cells[1].paragraphs[0]
        p_u.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        ru = p_u.add_run(f"({usn})")
        ru.font.name = "Times New Roman"
        ru.font.size = Pt(11)
        ru.font.bold = True
    for row in stud_table.rows:
        row.cells[0].width = Inches(2.6)
        row.cells[1].width = Inches(2.2)

    # EWIT Logo
    if EWIT_LOGO.exists():
        p_elogo = doc.add_paragraph()
        p_elogo.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_elogo.paragraph_format.space_before = Pt(8)
        p_elogo.paragraph_format.space_after = Pt(4)
        p_elogo.add_run().add_picture(str(EWIT_LOGO), width=Inches(1.1))

    # College Footer
    p_coll = doc.add_paragraph()
    p_coll.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_coll.paragraph_format.space_after = Pt(2)
    r_c1 = p_coll.add_run("EAST WEST INSTITUTE OF TECHNOLOGY\n")
    r_c1.font.name = "Times New Roman"
    r_c1.font.size = Pt(12)
    r_c1.font.bold = True

    r_c2 = p_coll.add_run("Off Magadi Main Road, Bengaluru, Karnataka — 560 091\n")
    r_c2.font.name = "Times New Roman"
    r_c2.font.size = Pt(11)

    r_c3 = p_coll.add_run("AY 2025–2026")
    r_c3.font.name = "Times New Roman"
    r_c3.font.size = Pt(11.5)
    r_c3.font.bold = True

    doc.add_page_break()

    # -------------------------------------------------------------
    # 2. VTU CERTIFICATE
    # -------------------------------------------------------------
    p_cert_title = doc.add_paragraph()
    p_cert_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_cert_title.paragraph_format.space_before = Pt(12)
    p_cert_title.paragraph_format.space_after = Pt(18)
    rc = p_cert_title.add_run("CERTIFICATE")
    rc.font.name = "Times New Roman"
    rc.font.size = Pt(16)
    rc.font.bold = True

    p_cert_txt = doc.add_paragraph()
    p_cert_txt.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p_cert_txt.paragraph_format.line_spacing = 1.5
    p_cert_txt.paragraph_format.space_after = Pt(18)
    p_cert_txt.add_run(
        "This is to certify that the project report entitled "
    )
    r_tbold = p_cert_txt.add_run("EMPLOYEE ATTRITION PREDICTION USING MACHINE LEARNING WITH EXPLAINABLE AI (XAI) ")
    r_tbold.font.bold = True
    p_cert_txt.add_run(
        "submitted by ANUSHA H G bearing the USN 1EW23IC006, CHARAN M S bearing the USN 1EW23IC012, "
        "NITHIN GOWDA R bearing the USN 1EW23IC031, and PUNARVA D N bearing the USN 1EW23IC037 to the "
        "Visvesvaraya Technological University, Belagavi, in partial fulfillment for the award of the degree of "
        "Bachelor of Engineering in CSE (IoT & Cybersecurity including Blockchain) is a bona fide record of project "
        "work carried out by them under my/our supervision. The contents of this report, in full or in parts, have not "
        "been submitted to any other Institution or University for the award of any degree."
    )

    doc.add_paragraph("\n\n")
    p_sig1 = doc.add_paragraph()
    p_sig1.paragraph_format.line_spacing = 1.15
    r_g1 = p_sig1.add_run("Signature\nName of the Guide\nProf. P Bharath Krishna\nDepartment of CSE (IoT & CS)")
    r_g1.font.size = Pt(11)

    p_sig2 = doc.add_paragraph()
    p_sig2.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    p_sig2.paragraph_format.line_spacing = 1.15
    r_h1 = p_sig2.add_run("Signature of HOD with seal\nDr. Venkatesha G\nHead of Department, CSE (IoT & CS)")
    r_h1.font.size = Pt(11)

    doc.add_page_break()

    # -------------------------------------------------------------
    # 3. DECLARATION
    # -------------------------------------------------------------
    p_dec_title = doc.add_paragraph()
    p_dec_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_dec_title.paragraph_format.space_before = Pt(12)
    p_dec_title.paragraph_format.space_after = Pt(18)
    rd = p_dec_title.add_run("DECLARATION")
    rd.font.name = "Times New Roman"
    rd.font.size = Pt(16)
    rd.font.bold = True

    p_dec_txt = doc.add_paragraph()
    p_dec_txt.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p_dec_txt.paragraph_format.line_spacing = 1.5
    p_dec_txt.paragraph_format.space_after = Pt(20)
    p_dec_txt.add_run(
        "I/We declare that this project report titled "
    )
    p_dec_txt.add_run("EMPLOYEE ATTRITION PREDICTION USING MACHINE LEARNING WITH EXPLAINABLE AI (XAI) ").font.bold = True
    p_dec_txt.add_run(
        "submitted in partial fulfillment of the degree of Bachelor of Engineering in CSE (IoT & Cybersecurity including Blockchain) "
        "is a record of original work carried out by me/us under the supervision of Prof. P Bharath Krishna, Dept of CSE (IoT & CyS Incl. BCT) "
        "and Dr. Venkatesha G, HOD, and has not formed the basis for the award of any other degree, in this or any other Institution or University. "
        "In keeping with ethical practice in reporting scientific information, due acknowledgements have been made wherever the findings of others have been cited."
    )

    p_dec_sign = doc.add_paragraph()
    p_dec_sign.paragraph_format.space_after = Pt(12)
    p_dec_sign.add_run("Signature of the student(s):").font.bold = True

    dec_table = doc.add_table(rows=5, cols=3)
    dec_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    dec_table.rows[0].cells[0].paragraphs[0].add_run("Sl. No.").font.bold = True
    dec_table.rows[0].cells[1].paragraphs[0].add_run("Name of Student & USN").font.bold = True
    dec_table.rows[0].cells[2].paragraphs[0].add_run("Signature").font.bold = True
    for idx, (sname, usn) in enumerate(students):
        r = dec_table.rows[idx + 1]
        r.cells[0].paragraphs[0].add_run(f"{idx+1}")
        r.cells[1].paragraphs[0].add_run(f"{sname} ({usn})")
        r.cells[2].paragraphs[0].add_run("_______________________")
    style_table(dec_table, [0.8, 3.8, 2.0], header_bg="475569")

    doc.add_page_break()

    # -------------------------------------------------------------
    # 4. COLLEGE CERTIFICATE (EWIT Format from Zero Trust Report)
    # -------------------------------------------------------------
    p_c1 = doc.add_paragraph()
    p_c1.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_c1.paragraph_format.space_after = Pt(2)
    p_c1.add_run("EAST WEST INSTITUTE OF TECHNOLOGY\n").font.bold = True
    p_c1.add_run("Off Magadi Main Road, Bengaluru, Karnataka — 560 091\n")
    p_c1.add_run("Department of CSE (IoT & Cybersecurity including Blockchain)\n\n").font.bold = True

    if EWIT_LOGO.exists():
        p_clogo = doc.add_paragraph()
        p_clogo.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_clogo.paragraph_format.space_after = Pt(8)
        p_clogo.add_run().add_picture(str(EWIT_LOGO), width=Inches(1.0))

    p_c2 = doc.add_paragraph()
    p_c2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_c2.paragraph_format.space_after = Pt(14)
    rc2 = p_c2.add_run("CERTIFICATE")
    rc2.font.name = "Times New Roman"
    rc2.font.size = Pt(16)
    rc2.font.bold = True

    p_ctxt = doc.add_paragraph()
    p_ctxt.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p_ctxt.paragraph_format.line_spacing = 1.5
    p_ctxt.paragraph_format.space_after = Pt(16)
    p_ctxt.add_run(
        "Certified that the project work entitled "
    )
    p_ctxt.add_run("EMPLOYEE ATTRITION PREDICTION USING MACHINE LEARNING WITH EXPLAINABLE AI (XAI) ").font.bold = True
    p_ctxt.add_run(
        "was carried out by ANUSHA H G (1EW23IC006), CHARAN M S (1EW23IC012), NITHIN GOWDA R (1EW23IC031), PUNARVA D N (1EW23IC037), "
        "bona fide students of East West Institute of Technology in partial fulfillment for the award of Bachelor of Engineering in "
        "CSE (IoT & Cybersecurity including Blockchain) of Visvesvaraya Technological University, Belagavi during the year 2025–2026. "
        "It is certified that all corrections/suggestions indicated for the Internal Assessment have been incorporated in the Report deposited "
        "in the departmental library. The project report has been approved as it satisfies the academic requirements in respect of Project work "
        "prescribed for the said Degree."
    )

    doc.add_paragraph("\n")
    sig_tab = doc.add_table(rows=2, cols=3)
    sig_tab.alignment = WD_TABLE_ALIGNMENT.CENTER
    sig_tab.rows[0].cells[0].paragraphs[0].add_run("Signature of Guide\n(Prof. P Bharath Krishna)")
    sig_tab.rows[0].cells[1].paragraphs[0].add_run("Signature of HOD\n(Dr. Venkatesha G)")
    sig_tab.rows[0].cells[2].paragraphs[0].add_run("Signature of Principal\n(Dr. Nagaraj G. Cholli)")
    for cell in sig_tab.rows[0].cells:
        cell.paragraphs[0].runs[0].font.size = Pt(10)
        cell.paragraphs[0].runs[0].font.bold = True
    for cell in sig_tab.rows[1].cells:
        cell.paragraphs[0].add_run("\n\nDate: _______________")

    doc.add_paragraph("\n")
    p_ext = doc.add_paragraph()
    p_ext.add_run("External Viva Examiners:").font.bold = True

    viva_tab = doc.add_table(rows=3, cols=3)
    viva_tab.alignment = WD_TABLE_ALIGNMENT.CENTER
    viva_tab.rows[0].cells[0].paragraphs[0].add_run("Sl. No.").font.bold = True
    viva_tab.rows[0].cells[1].paragraphs[0].add_run("Name of the Examiners").font.bold = True
    viva_tab.rows[0].cells[2].paragraphs[0].add_run("Signature with Date").font.bold = True
    viva_tab.rows[1].cells[0].paragraphs[0].add_run("1.")
    viva_tab.rows[1].cells[1].paragraphs[0].add_run("Internal Examiner: _______________________")
    viva_tab.rows[1].cells[2].paragraphs[0].add_run("___________________________")
    viva_tab.rows[2].cells[0].paragraphs[0].add_run("2.")
    viva_tab.rows[2].cells[1].paragraphs[0].add_run("External Examiner: _______________________")
    viva_tab.rows[2].cells[2].paragraphs[0].add_run("___________________________")
    style_table(viva_tab, [0.8, 3.8, 2.0], header_bg="475569")

    doc.add_page_break()

    # -------------------------------------------------------------
    # 5. ABSTRACT & KEYWORDS
    # -------------------------------------------------------------
    p_abs = doc.add_paragraph()
    p_abs.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_abs.paragraph_format.space_before = Pt(12)
    p_abs.paragraph_format.space_after = Pt(16)
    ra = p_abs.add_run("ABSTRACT")
    ra.font.name = "Times New Roman"
    ra.font.size = Pt(16)
    ra.font.bold = True

    p_atxt = doc.add_paragraph()
    p_atxt.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p_atxt.paragraph_format.line_spacing = 1.5
    p_atxt.add_run(
        "Voluntary employee turnover poses substantial operational disruptions and severe financial losses to contemporary knowledge-driven "
        "organizations, where recruiting, onboarding, and domain knowledge replacement costs range between 50% and 200% of an employee's annual salary. "
        "This project presents an enterprise-ready, explainable classification-based Machine Learning pipeline to identify at-risk personnel and diagnose "
        "underlying attrition drivers. Built on the IBM HR Analytics dataset (1,470 employee records, 35 attributes), the system addresses the natural "
        "class imbalance (83.9% retained vs. 16.1% departed) by applying cost-sensitive balanced weighting, stratified 5-fold cross-validation, and "
        "optimizing the F1-score rather than raw accuracy, completely resolving the Accuracy Paradox."
    )

    p_atxt2 = doc.add_paragraph()
    p_atxt2.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p_atxt2.paragraph_format.line_spacing = 1.5
    p_atxt2.add_run(
        "Five supervised classification algorithms—Logistic Regression, Decision Tree, Random Forest, K-Nearest Neighbors (KNN), and Support Vector "
        "Machine (SVM)—were trained within a strict scikit-learn Pipeline and ColumnTransformer to eliminate data leakage. Support Vector Machine with an RBF "
        "kernel emerged as the champion classifier, achieving a cross-validation F1-score of 0.5532 and test ROC-AUC of 0.8168. Tuning the decision threshold "
        "to 0.33 maximized the hold-out test F1-score to 0.5594, detecting 56.34% of real leavers with 55.56% precision and 85.71% overall accuracy. "
        "To ensure human-in-the-loop auditability for sensitive HR decisions, the architecture integrates Dual-Layer Explainability: Permutation Importance "
        "for global factor attribution and an accelerated SHAP KernelExplainer (optimized from 57s to 0.32s via k-means clustering) for local case "
        "diagnostics. The complete solution is deployed as a production-grade Streamlit web application supporting interactive persona presets, single-employee "
        "diagnostics, and workforce-scale batch CSV scoring with one-click report exports."
    )

    p_kw = doc.add_paragraph()
    p_kw.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p_kw.paragraph_format.space_before = Pt(12)
    p_kw.add_run("Keywords: ").font.bold = True
    p_kw.add_run(
        "Employee Attrition Prediction, Machine Learning, Class Imbalance, Support Vector Machines, Random Forest, "
        "Accuracy Paradox, Cost-Sensitive Learning, Permutation Importance, SHAP (Shapley Additive Explanations), Streamlit, Human Resource Analytics"
    )

    doc.add_page_break()

    # -------------------------------------------------------------
    # 6. ACKNOWLEDGEMENT
    # -------------------------------------------------------------
    p_ack = doc.add_paragraph()
    p_ack.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_ack.paragraph_format.space_before = Pt(12)
    p_ack.paragraph_format.space_after = Pt(16)
    rack = p_ack.add_run("ACKNOWLEDGEMENT")
    rack.font.name = "Times New Roman"
    rack.font.size = Pt(16)
    rack.font.bold = True

    p_ack_txt1 = doc.add_paragraph()
    p_ack_txt1.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p_ack_txt1.paragraph_format.line_spacing = 1.5
    p_ack_txt1.add_run(
        "With great pride we convey our gratitude and appreciation to our alma mater “East West Institute of Technology” for giving us "
        "the required platform and academic infrastructure for the fulfillment of the Project on “Employee Attrition Prediction Using Machine Learning "
        "with Explainable AI (XAI)” as per the V.T.U requirements."
    )

    p_ack_txt2 = doc.add_paragraph()
    p_ack_txt2.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p_ack_txt2.paragraph_format.line_spacing = 1.5
    p_ack_txt2.add_run(
        "We express our sincere thanks to Dr. Nagaraj G. Cholli, Principal of East West Institute of Technology, for providing us with an "
        "excellent educational environment and laboratory facilities to complete this project."
    )

    p_ack_txt3 = doc.add_paragraph()
    p_ack_txt3.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p_ack_txt3.paragraph_format.line_spacing = 1.5
    p_ack_txt3.add_run(
        "We express wholehearted gratitude to Dr. Venkatesha G, respectable HOD of the Department of CSE (IoT & Cybersecurity including Blockchain). "
        "We wish to acknowledge his constant guidance, administrative support, and encouragement throughout the project."
    )

    p_ack_txt4 = doc.add_paragraph()
    p_ack_txt4.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p_ack_txt4.paragraph_format.line_spacing = 1.5
    p_ack_txt4.add_run(
        "It is our pleasure to express our deepest gratitude to our project guide and Course Instructor, Prof. P Bharath Krishna, Assistant Professor, "
        "Department of CSE (IoT & CS), for his continuous technical mentorship, constructive suggestions, and valuable advice from the inception "
        "of the project till its completion, without which this work would not have been accomplished."
    )

    p_ack_txt5 = doc.add_paragraph()
    p_ack_txt5.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p_ack_txt5.paragraph_format.line_spacing = 1.5
    p_ack_txt5.add_run(
        "Last but not the least, we express our heartfelt thanks to all the faculty and non-teaching staff of the Department of CSE (IoT & CS including BT), "
        "and to our families and friends who extended their constant moral support."
    )

    doc.add_page_break()

    # -------------------------------------------------------------
    # 7. LIST OF TABLES & LIST OF FIGURES
    # -------------------------------------------------------------
    p_lt = doc.add_paragraph()
    p_lt.paragraph_format.space_before = Pt(10)
    p_lt.paragraph_format.space_after = Pt(8)
    rlt = p_lt.add_run("LIST OF TABLES")
    rlt.font.size = Pt(14)
    rlt.font.bold = True

    lt_tab = doc.add_table(rows=9, cols=3)
    lt_tab.alignment = WD_TABLE_ALIGNMENT.CENTER
    lt_tab.rows[0].cells[0].paragraphs[0].add_run("Table No.").font.bold = True
    lt_tab.rows[0].cells[1].paragraphs[0].add_run("Title of Table").font.bold = True
    lt_tab.rows[0].cells[2].paragraphs[0].add_run("Page No.").font.bold = True

    tables_data = [
        ("Table 1.1", "Comparison of Classical Rule-Based Retention vs ML Risk Analytics", "3"),
        ("Table 2.1", "Comparative Study of Related Works in Predictive HR Analytics", "6"),
        ("Table 3.1", "Hardware Requirements Specification", "8"),
        ("Table 3.2", "Software Requirements Specification", "9"),
        ("Table 4.1", "IBM HR Analytics Attribute Schema & Transformation Specification", "13"),
        ("Table 6.1", "Technology Stack Matrix & Computational Libraries", "18"),
        ("Table 7.1", "Hold-Out Test Performance Benchmark Across 5 ML Classifiers", "21"),
        ("Table 7.2", "Hyperparameter Tuning Search Grids and Optimal Configurations", "23"),
    ]
    for idx, (tno, ttitle, pg) in enumerate(tables_data):
        r = lt_tab.rows[idx + 1]
        r.cells[0].paragraphs[0].add_run(tno)
        r.cells[1].paragraphs[0].add_run(ttitle)
        r.cells[2].paragraphs[0].add_run(pg)
    style_table(lt_tab, [1.2, 4.4, 1.0], header_bg="475569")

    doc.add_paragraph("\n")
    p_lf = doc.add_paragraph()
    p_lf.paragraph_format.space_before = Pt(10)
    p_lf.paragraph_format.space_after = Pt(8)
    rlf = p_lf.add_run("LIST OF FIGURES")
    rlf.font.size = Pt(14)
    rlf.font.bold = True

    lf_tab = doc.add_table(rows=11, cols=3)
    lf_tab.alignment = WD_TABLE_ALIGNMENT.CENTER
    lf_tab.rows[0].cells[0].paragraphs[0].add_run("Figure No.").font.bold = True
    lf_tab.rows[0].cells[1].paragraphs[0].add_run("Title of Figure").font.bold = True
    lf_tab.rows[0].cells[2].paragraphs[0].add_run("Page No.").font.bold = True

    figures_data = [
        ("Figure 3.1", "Attrition Class Frequencies and Proportions (83.9% vs. 16.1%)", "10"),
        ("Figure 3.2", "Attrition Disparities by OverTime Status and Marital Status", "11"),
        ("Figure 3.3", "Attrition Percentages Across Specific Organizational Job Roles", "11"),
        ("Figure 3.4", "Attrition Trends across Satisfaction Dimensions (Likert Scales)", "12"),
        ("Figure 3.5", "Distributions of Continuous Drivers (Age, Income, Distance, Tenure)", "12"),
        ("Figure 3.6", "Correlation Heatmap: Numerical Features vs. Attrition Target", "13"),
        ("Figure 4.1", "End-to-End Leak-Free Pipeline & Architecture Diagram", "15"),
        ("Figure 7.1", "Comparative Model Performance Bar Chart (F1-Score)", "22"),
        ("Figure 7.2", "ROC Curves Comparison Across 5 Machine Learning Classifiers", "22"),
        ("Figure 7.3", "Confusion Matrix Heatmap of Champion SVM at Optimal Threshold 0.33", "24"),
    ]
    for idx, (fno, ftitle, pg) in enumerate(figures_data):
        r = lf_tab.rows[idx + 1]
        r.cells[0].paragraphs[0].add_run(fno)
        r.cells[1].paragraphs[0].add_run(ftitle)
        r.cells[2].paragraphs[0].add_run(pg)
    style_table(lf_tab, [1.2, 4.4, 1.0], header_bg="475569")

    doc.add_page_break()

    # -------------------------------------------------------------
    # 8. CONTENTS (TABLE OF CONTENTS)
    # -------------------------------------------------------------
    p_toc = doc.add_paragraph()
    p_toc.paragraph_format.space_before = Pt(10)
    p_toc.paragraph_format.space_after = Pt(8)
    rtoc = p_toc.add_run("CONTENTS")
    rtoc.font.size = Pt(14)
    rtoc.font.bold = True

    toc_tab = doc.add_table(rows=17, cols=3)
    toc_tab.alignment = WD_TABLE_ALIGNMENT.CENTER
    toc_tab.rows[0].cells[0].paragraphs[0].add_run("Chapter No.").font.bold = True
    toc_tab.rows[0].cells[1].paragraphs[0].add_run("Chapter / Section Title").font.bold = True
    toc_tab.rows[0].cells[2].paragraphs[0].add_run("Page No.").font.bold = True

    toc_entries = [
        ("—", "Certificate & Declaration", "ii"),
        ("—", "Abstract & Acknowledgement", "iv"),
        ("—", "List of Tables & Figures", "vi"),
        ("Chapter 1", "Introduction & Problem Formulation", "1"),
        ("Chapter 2", "Literature Survey", "4"),
        ("Chapter 3", "System Requirements Specification", "7"),
        ("Chapter 4", "System Design & Pipeline Architecture", "10"),
        ("Chapter 5", "Methodology / ML Model Development", "14"),
        ("Chapter 6", "Implementation Details", "17"),
        ("Chapter 7", "Testing, Results and Performance Evaluation", "20"),
        ("Chapter 8", "Conclusion and Future Scope", "26"),
        ("—", "References", "28"),
        ("Appendix A", "System Snapshots & User Interface", "30"),
        ("Appendix B", "Source Code Directory Structure", "33"),
        ("Appendix C", "Automated Unit Test Suite & Audit Logs", "35"),
        ("Appendix D", "Guide Approval & Evaluation Marks Sheet", "37"),
    ]
    for idx, (cno, ctitle, pg) in enumerate(toc_entries):
        r = toc_tab.rows[idx + 1]
        r.cells[0].paragraphs[0].add_run(cno)
        r.cells[1].paragraphs[0].add_run(ctitle)
        r.cells[2].paragraphs[0].add_run(pg)
    style_table(toc_tab, [1.3, 4.3, 1.0], header_bg="1F3A5F")

    doc.add_page_break()

    # -------------------------------------------------------------
    # PROJECT INFORMATION & SPECIFICATIONS (VTU Standard from Zero Trust)
    # -------------------------------------------------------------
    p_info = doc.add_paragraph()
    p_info.paragraph_format.space_before = Pt(8)
    p_info.paragraph_format.space_after = Pt(4)
    r_inf = p_info.add_run("PROJECT INFORMATION & SPECIFICATIONS")
    r_inf.font.size = Pt(13)
    r_inf.font.bold = True

    info_tab = doc.add_table(rows=6, cols=2)
    info_tab.alignment = WD_TABLE_ALIGNMENT.CENTER
    info_data = [
        ("Project Title", "Employee Attrition Prediction Using Machine Learning With Explainable AI (XAI)"),
        ("Degree & Department", "B.E. in CSE (IoT & Cybersecurity including Blockchain), East West Institute of Technology"),
        ("University & Curriculum", "Visvesvaraya Technological University (VTU), Machine Learning [BIC703], AY 2025–2026"),
        ("Student Team", "ANUSHA H G (1EW23IC006), CHARAN M S (1EW23IC012), NITHIN GOWDA R (1EW23IC031), PUNARVA D N (1EW23IC037)"),
        ("Guide & HOD", "Prof. P Bharath Krishna (Course Instructor & Guide), Dr. Venkatesha G (HOD, Dept of CSE)"),
        ("Evaluation Rubrics Marks", "100 Marks (Phase 1: 40 Marks, Phase 2: 60 Marks, Downscaled to 10 Marks for CIE)"),
    ]
    for idx, (k, v) in enumerate(info_data):
        r = info_tab.rows[idx]
        r.cells[0].paragraphs[0].add_run(k).font.bold = True
        r.cells[1].paragraphs[0].add_run(v)
    style_table(info_tab, [2.2, 4.4], header_bg="475569")

    doc.add_paragraph("\n")
    p_form = doc.add_paragraph()
    p_form.paragraph_format.space_before = Pt(6)
    p_form.paragraph_format.space_after = Pt(4)
    r_frm = p_form.add_run("REPORT FORMATTING SPECIFICATIONS (VTU GUIDELINES)")
    r_frm.font.size = Pt(13)
    r_frm.font.bold = True

    form_tab = doc.add_table(rows=6, cols=2)
    form_tab.alignment = WD_TABLE_ALIGNMENT.CENTER
    form_data = [
        ("Paper Size & Margins", "A4 (8.27\" x 11.69\"); Left: 1.25\", Right: 1.0\", Top: 1.0\", Bottom: 1.0\""),
        ("Font Family & Body Text", "Times New Roman; Regular 12pt, 1.5 line spacing, Justified alignment"),
        ("Chapter Title", "Times New Roman; 16pt Bold, Uppercase, Centered (e.g., CHAPTER 1)"),
        ("Major Headings (H1 / H2)", "Times New Roman; 13.5pt Bold, Uppercase (e.g., 1.1 BACKGROUND AND MOTIVATION)"),
        ("Sub-Headings (H3)", "Times New Roman; 12pt Bold (e.g., 3.1.1 Authentication & Enrollment Module)"),
        ("Tables & Captions", "Times New Roman; 9.5pt/10pt with shaded header row; Figures captioned in 9.5pt Italic"),
    ]
    for idx, (k, v) in enumerate(form_data):
        r = form_tab.rows[idx]
        r.cells[0].paragraphs[0].add_run(k).font.bold = True
        r.cells[1].paragraphs[0].add_run(v)
    style_table(form_tab, [2.2, 4.4], header_bg="475569")

    doc.add_page_break()

    # -------------------------------------------------------------
    # CHAPTER 1: INTRODUCTION
    # -------------------------------------------------------------
    p_c1 = doc.add_paragraph()
    p_c1.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_c1.paragraph_format.space_before = Pt(10)
    p_c1.paragraph_format.space_after = Pt(4)
    rc1 = p_c1.add_run("CHAPTER 1\nINTRODUCTION")
    rc1.font.name = "Times New Roman"
    rc1.font.size = Pt(16)
    rc1.font.bold = True

    doc.add_paragraph().add_run("1.1 BACKGROUND AND MOTIVATION").font.bold = True
    doc.add_paragraph(
        "Human capital represents the primary driver of innovation and operational resilience in contemporary knowledge-based organizations. "
        "Every enterprise relies on retaining specialized engineering, managerial, and domain talent to maintain competitiveness. "
        "However, voluntary employee departure poses profound organizational vulnerabilities. When high-performing staff leave unexpectedly, "
        "enterprises incur enormous direct and indirect costs, including recruiter fees, signing bonuses, candidate interview overhead, and "
        "prolonged productivity disruption."
    )
    doc.add_paragraph(
        "According to industrial benchmarks from the Society for Human Resource Management (SHRM), replacing a professional employee costs between "
        "50% and 200% of their annual salary. In technical and specialized roles, the knowledge loss can destabilize critical engineering pipelines. "
        "Historically, corporate human resource management has relied on reactive exit interviews and static, rule-based annual appraisals. These "
        "mechanisms operate post-facto—after an employee has already tendered resignation—offering zero opportunity for timely retention interventions."
    )
    doc.add_paragraph(
        "The emergence of Machine Learning (ML) and predictive people analytics offers an unprecedented opportunity to fundamentally re-engineer "
        "workforce retention. By learning complex multi-dimensional patterns across employee demographics, commute distance, compensation tiers, "
        "overtime workloads, and satisfaction metrics, machine learning classifiers can identify early attrition risk signals weeks before resignation occurs. "
        "Furthermore, integrating Explainable Artificial Intelligence (XAI) ensures that predictions are accompanied by human-interpretable feature "
        "attributions, empowering HR managers to design targeted, personalized retention programs."
    )

    doc.add_paragraph().add_run("1.2 PROBLEM STATEMENT").font.bold = True
    doc.add_paragraph(
        "Traditional human resource management systems suffer from severe and well-documented operational weaknesses:\n"
        "• Reactive Assessment: Exit interviews take place after resignation, precluding retention interventions.\n"
        "• The Accuracy Paradox: In real-world workforces, attrition is rare (~16%), causing naive models to score deceptively high accuracy (~84%) "
        "while failing to detect actual departing employees (0% recall).\n"
        "• Black-Box Predictions: Standard complex algorithms output raw probabilities without explaining why an employee is at risk, preventing "
        "managers from knowing whether to adjust salary, reduce overtime, or address commute friction.\n"
        "• Data Leakage: Flawed ML pipelines frequently fit scalers and encoders across the entire dataset prior to splitting, corrupting model validity."
    )

    doc.add_paragraph().add_run("1.3 OBJECTIVES").font.bold = True
    doc.add_paragraph().add_run("Phase I Objectives (Semester VI / Academic Phase 1):").font.bold = True
    doc.add_paragraph(
        "• 1. Literature Survey: Conduct a comprehensive review of machine learning applications in predictive talent analytics and class imbalance handling.\n"
        "• 2. Dataset Hygiene & Dictionary: Ingest the IBM HR Analytics dataset, verify data integrity (0 missing values, 0 duplicates), and compile an exhaustive data dictionary.\n"
        "• 3. Exploratory Data Analysis: Perform univariate, bivariate, and multivariate visual analytics (minimum 5 visualizations) to identify primary churn drivers.\n"
        "• 4. Preprocessing Pipeline: Construct a scikit-learn ColumnTransformer and Pipeline architecture to eliminate data leakage.\n"
        "• 5. Baseline Modeling: Develop and evaluate 5 initial classifiers (SVM, RF, LR, DT, KNN) under stratified cross-validation."
    )
    doc.add_paragraph().add_run("Phase II Objectives (Semester VII / Academic Phase 2):").font.bold = True
    doc.add_paragraph(
        "• 1. Hyperparameter Tuning: Execute 5-fold Stratified GridSearchCV/RandomizedSearchCV across all 5 models optimizing the F1-score.\n"
        "• 2. Decision Threshold Calibration: Tune the decision boundary from 0.50 to the empirical optimum (0.33) using Precision-Recall curve analysis.\n"
        "• 3. Accelerated Local XAI: Implement and optimize SHAP KernelExplainer from 57 seconds down to 0.32 seconds using k-means centroid clustering.\n"
        "• 4. Full Web Deployment: Build an interactive Streamlit application supporting single-profile diagnostics, persona presets, and workforce batch CSV scoring.\n"
        "• 5. Automated Verification: Develop an automated unit test suite (10/10 tests) and compile the complete VTU dissertation report."
    )

    doc.add_paragraph().add_run("1.4 SCOPE OF THE PROJECT").font.bold = True
    doc.add_paragraph(
        "The project encompasses data ingestion, exploratory visual analytics, mathematical preprocessing, cross-validated model training, "
        "hyperparameter optimization, threshold calibration, XAI explainability, automated unit testing, and web deployment. It serves as an "
        "enterprise-ready template for data-driven human resource decision support."
    )

    doc.add_paragraph().add_run("1.5 ORGANISATION OF THE REPORT").font.bold = True
    doc.add_paragraph(
        "The remainder of this report is organized as follows: Chapter 2 surveys existing literature in predictive talent analytics and imbalance handling. "
        "Chapter 3 defines functional, non-functional, hardware, and software requirements. Chapter 4 details system architecture, preprocessing design, and "
        "data flows. Chapter 5 formalizes the research methodology, ML algorithms, and XAI mathematics. Chapter 6 describes technical implementation details. "
        "Chapter 7 presents empirical results, benchmarks, confusion matrices, and economic cost analyses. Chapter 8 concludes the report with future scope."
    )

    doc.add_page_break()

    # -------------------------------------------------------------
    # CHAPTER 2: LITERATURE SURVEY
    # -------------------------------------------------------------
    p_c2 = doc.add_paragraph()
    p_c2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_c2.paragraph_format.space_before = Pt(10)
    p_c2.paragraph_format.space_after = Pt(4)
    rc2 = p_c2.add_run("CHAPTER 2\nLITERATURE SURVEY")
    rc2.font.name = "Times New Roman"
    rc2.font.size = Pt(16)
    rc2.font.bold = True

    doc.add_paragraph().add_run("2.1 CLASSICAL PERIMETER & RULE-BASED HR RETENTION SYSTEMS").font.bold = True
    doc.add_paragraph(
        "Early talent management systems relied almost exclusively on retrospective surveys and heuristic scorecards. Anderson et al. [1] "
        "and industrial case studies proved that manual exit interviews suffer from severe reporting bias: departing employees frequently cite "
        "neutral reasons to avoid burning bridges with former managers, obscuring true grievances such as chronic overtime or lack of promotion. "
        "Moreover, static threshold rules (e.g., flagging any employee with low annual appraisal) fail to capture the complex, multi-variable interactions "
        "that precede departure."
    )

    doc.add_paragraph().add_run("2.2 MACHINE LEARNING IN TALENT ANALYTICS & CHURN PREDICTION").font.bold = True
    doc.add_paragraph(
        "The transition to predictive modeling has gained significant academic traction. IBM Watson Analytics [3] published the foundational HR "
        "benchmark dataset, demonstrating that tree ensembles and linear margins could capture subtle tenure and compensation interactions. Breiman [5] "
        "demonstrated that Random Forests, through bagging and random feature subsampling, provide high resilience against non-linear data distributions. "
        "However, early applications often neglected the critical challenge of class imbalance, erroneously reporting 85%+ accuracy while detecting "
        "virtually no real leavers."
    )

    doc.add_paragraph().add_run("2.3 CLASS IMBALANCE & COST-SENSITIVE LEARNING").font.bold = True
    doc.add_paragraph(
        "In typical corporate environments, attrition rates range between 10% and 20% annually. Chawla et al. [4] introduced the Synthetic Minority "
        "Oversampling Technique (SMOTE) to synthetically balance minority distributions. However, recent literature indicates that in high-dimensional "
        "tabular datasets with mixed categorical and numerical features, synthetic oversampling often introduces noisy artifacts into sparse regions. "
        "Cost-sensitive learning, which weights the loss function inversely proportional to class frequencies, offers a mathematically rigorous, "
        "zero-latency alternative that avoids synthesizing artificial records."
    )

    doc.add_paragraph().add_run("2.4 EXPLAINABLE ARTIFICIAL INTELLIGENCE (XAI) & SHAP").font.bold = True
    doc.add_paragraph(
        "Deploying black-box predictive models in human capital management introduces severe ethical and practical barriers. Lundberg and Lee [1] "
        "formalized SHAP (Shapley Additive Explanations), grounding local feature attribution in cooperative game theory. By allocating fair credit "
        "to each feature for shifting the model's base probability, SHAP bridges the gap between complex non-linear classification and actionable "
        "managerial decision support."
    )

    doc.add_paragraph().add_run("2.5 COMPARATIVE STUDY OF RELATED WORKS").font.bold = True
    doc.add_paragraph("Table 2.1 presents a comparative study contrasting prior published literature against the proposed system architecture:")

    rel_table = doc.add_table(rows=5, cols=5)
    rel_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    rel_headers = ["Authors & Reference", "Algorithm(s)", "Imbalance Strategy", "Explainability (XAI)", "Key Limitation"]
    for j, h in enumerate(rel_headers):
        rel_table.rows[0].cells[j].paragraphs[0].add_run(h)
    
    rel_rows = [
        ("IBM Watson (2016) [3]", "Logistic Regression, Decision Trees", "None (Raw Accuracy)", "None (Black Box)", "Suffers from Accuracy Paradox; low recall on leavers"),
        ("Chawla et al. (2002) [4]", "C4.5 Decision Tree + SMOTE", "Synthetic Oversampling", "Tree rules only", "SMOTE generates unrealistic synthetic outliers in mixed data"),
        ("Breiman et al. (2001) [5]", "Random Forest Regressor", "Subsampling / Bagging", "MDI Impurity", "MDI importance is biased toward high-cardinality features"),
        ("Proposed Architecture", "Tuned SVM (RBF) + RF + LR + DT + KNN", "Balanced Loss + 0.33 Threshold", "Dual-Layer: Permutation + Fast SHAP", "Snapshot dataset requires longitudinal survival extension"),
    ]
    for idx, rvals in enumerate(rel_rows):
        r = rel_table.rows[idx + 1]
        for col_idx, val in enumerate(rvals):
            r.cells[col_idx].paragraphs[0].add_run(val)
    style_table(rel_table, [1.5, 1.4, 1.2, 1.2, 1.5], header_bg="1F3A5F")

    doc.add_page_break()

    # -------------------------------------------------------------
    # CHAPTER 3: SYSTEM REQUIREMENTS
    # -------------------------------------------------------------
    p_c3 = doc.add_paragraph()
    p_c3.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_c3.paragraph_format.space_before = Pt(10)
    p_c3.paragraph_format.space_after = Pt(4)
    rc3 = p_c3.add_run("CHAPTER 3\nSYSTEM REQUIREMENTS SPECIFICATION")
    rc3.font.name = "Times New Roman"
    rc3.font.size = Pt(16)
    rc3.font.bold = True

    doc.add_paragraph().add_run("3.1 FUNCTIONAL REQUIREMENTS").font.bold = True
    doc.add_paragraph(
        "• Data Ingestion & Preprocessing: Ingests raw CSV data, removes 4 non-informative attributes, applies StandardScaler on 23 numerical features "
        "and OneHotEncoder on 7 categorical features.\n"
        "• Model Benchmark & Training: Trains and compares 5 distinct classification models under 5-fold stratified cross-validation.\n"
        "• Hyperparameter Tuning: Executes grid and randomized searches across parameter spaces, selecting the best model by cross-validated F1-score.\n"
        "• Decision Threshold Optimization: Allows threshold tuning (defaulting to 0.33) to maximize recall and F1-score.\n"
        "• Fast SHAP Local Explainability: Generates Shapley feature attributions within 0.5 seconds using background k-means summary.\n"
        "• Interactive Dashboard: Provides a multi-page Streamlit application with individual profile presets and workforce batch CSV scoring."
    )

    doc.add_paragraph().add_run("3.2 NON-FUNCTIONAL REQUIREMENTS").font.bold = True
    doc.add_paragraph(
        "• Performance & Latency: Single-prediction inference executes in under 20 ms; SHAP local attribution executes in under 500 ms.\n"
        "• Security & Data Leakage Prevention: Preprocessing is fitted strictly on training sets; no test data statistics leak into transformers.\n"
        "• Reliability & Test Coverage: Pipeline integrity is validated with automated unit tests achieving 100% pass rate.\n"
        "• Portability: Cross-platform Python execution compatible with Windows, Linux, and macOS."
    )

    doc.add_paragraph().add_run("3.3 HARDWARE REQUIREMENTS SPECIFICATION").font.bold = True
    hw_table = doc.add_table(rows=5, cols=2)
    hw_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    hw_data = [
        ("Processor", "Intel Core i5 / AMD Ryzen 5 or higher (minimum 4 cores, 2.4 GHz)"),
        ("System Memory (RAM)", "8 GB DDR4 RAM minimum (16 GB recommended for grid searches)"),
        ("Storage", "500 MB free hard drive space for dataset, plots, model artifacts, and reports"),
        ("Display Resolution", "1366 x 768 minimum display resolution (1920 x 1080 Full HD recommended)"),
    ]
    for idx, (k, v) in enumerate(hw_data):
        r = hw_table.rows[idx]
        r.cells[0].paragraphs[0].add_run(k).font.bold = True
        r.cells[1].paragraphs[0].add_run(v)
    style_table(hw_table, [2.2, 4.4], header_bg="475569")

    doc.add_paragraph("\n")
    doc.add_paragraph().add_run("3.4 SOFTWARE REQUIREMENTS SPECIFICATION").font.bold = True
    sw_table = doc.add_table(rows=6, cols=2)
    sw_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    sw_data = [
        ("Operating System", "Microsoft Windows 10/11 (64-bit) or Ubuntu Linux 22.04 LTS"),
        ("Programming Language", "Python 3.10 to Python 3.14 (Validated on Python 3.14.3)"),
        ("Core ML & Data Libraries", "scikit-learn >= 1.5, pandas >= 2.2, numpy >= 2.0, scipy >= 1.11, joblib >= 1.3"),
        ("Explainability & Analytics", "shap >= 0.45 (Accelerated KernelExplainer & TreeExplainer)"),
        ("Visualization & Web App", "streamlit >= 1.36, plotly >= 5.20, matplotlib >= 3.8, seaborn >= 0.13"),
        ("Documentation & Formats", "python-docx >= 1.1, python-pptx >= 1.0, nbformat >= 5.9"),
    ]
    for idx, (k, v) in enumerate(sw_data):
        r = sw_table.rows[idx]
        r.cells[0].paragraphs[0].add_run(k).font.bold = True
        r.cells[1].paragraphs[0].add_run(v)
    style_table(sw_table, [2.2, 4.4], header_bg="475569")

    doc.add_page_break()

    # -------------------------------------------------------------
    # CHAPTER 4: SYSTEM DESIGN AND ARCHITECTURE
    # -------------------------------------------------------------
    p_c4 = doc.add_paragraph()
    p_c4.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_c4.paragraph_format.space_before = Pt(10)
    p_c4.paragraph_format.space_after = Pt(4)
    rc4 = p_c4.add_run("CHAPTER 4\nSYSTEM DESIGN AND ARCHITECTURE")
    rc4.font.name = "Times New Roman"
    rc4.font.size = Pt(16)
    rc4.font.bold = True

    doc.add_paragraph().add_run("4.1 HIGH-LEVEL ARCHITECTURE").font.bold = True
    doc.add_paragraph(
        "The system architecture is structured across four decoupled, modular layers:\n"
        "1. Ingestion & Preprocessing Layer: Cleans raw CSV records, removes constant features, and applies ColumnTransformer scaling.\n"
        "2. Predictive Modeling Layer: Houses the 5 benchmarked classifiers and the serialized champion Support Vector Machine pipeline.\n"
        "3. Dual-Layer Explainability Engine: Permutation Importance computes global feature drops; accelerated SHAP decomposes local samples.\n"
        "4. Interactive Presentation Layer: Streamlit web console featuring real-time risk gauges, persona presets, and workforce batch scoring."
    )

    doc.add_paragraph().add_run("4.2 DATA PREPROCESSING & LEAKAGE-FREE PIPELINE DESIGN").font.bold = True
    doc.add_paragraph(
        "Data leakage occurs when information from outside the training dataset is used to create the model. In tabular machine learning, "
        "computing global mean and standard deviation across the full dataset leaks test set distribution into the training phase. "
        "To mathematically guarantee zero data leakage, our architecture strictly couples transformers and classifiers inside a scikit-learn Pipeline:"
    )
    doc.add_paragraph(
        "Pipeline(steps=[\n"
        "    ('pre', ColumnTransformer(transformers=[\n"
        "        ('num', StandardScaler(), NUMERICAL_FEATURES),\n"
        "        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_FEATURES)\n"
        "    ])),\n"
        "    ('clf', SVC(C=0.5, gamma='scale', probability=True, class_weight='balanced'))\n"
        "])"
    )

    doc.add_paragraph().add_run("4.3 DATA FLOW ARCHITECTURE (DFD LEVEL 0 & LEVEL 1)").font.bold = True
    doc.add_paragraph(
        "The data flow of the Employee Attrition Analytics platform is modeled across two formal levels:\n"
        "• DFD Level 0 (Context Diagram): The HR Officer inputs raw employee attributes or CSV files. The Machine Learning Engine ingests "
        "the records, applies pipeline transformations, evaluates model risk against the calibrated threshold (0.33), computes SHAP attributions, "
        "and outputs the risk probability, categorical verdict, and top risk factors.\n"
        "• DFD Level 1 (System Decomposition): Inbound data flows through Process 1.0 (Validation & Cleaning), Process 2.0 (ColumnTransformer "
        "Scaling & Encoding), Process 3.0 (Model Probability Estimation), Process 4.0 (Threshold & Risk Tier Classification), and Process 5.0 "
        "(SHAP Attribution & Report Generation)."
    )

    doc.add_paragraph().add_run("4.4 ATTRIBUTE SCHEMA & PREPROCESSING SPECIFICATION").font.bold = True
    doc.add_paragraph("Table 4.1 specifies the attribute schema and preprocessing transformations across the 35 features:")

    sch_tab = doc.add_table(rows=11, cols=4)
    sch_tab.alignment = WD_TABLE_ALIGNMENT.CENTER
    sch_headers = ["Attribute Group", "Features Count", "Input Data Types", "Pipeline Transformation Applied"]
    for j, h in enumerate(sch_headers):
        sch_tab.rows[0].cells[j].paragraphs[0].add_run(h)

    sch_rows = [
        ("Target Variable", "1 Feature", "String (Yes / No)", "Mapped to Binary Integer (0 = No, 1 = Yes)"),
        ("Demographics", "5 Features", "Integer, Nominal String", "Age/Distance scaled via StandardScaler; Gender/Marital One-Hot Encoded"),
        ("Job & Organizational", "8 Features", "Nominal String, Ordinal Int", "JobRole/Department One-Hot Encoded; JobLevel scaled"),
        ("Compensation & Financial", "6 Features", "Continuous Integer", "MonthlyIncome, Rates, Hike, StockOption scaled via StandardScaler"),
        ("Satisfaction Metrics", "5 Features", "Ordinal Likert (1 to 4)", "Scaled via StandardScaler; evaluated as continuous signals"),
        ("Career Tenure & History", "6 Features", "Continuous Integer", "TotalWorkingYears, Tenure at company/role scaled via StandardScaler"),
        ("Removed (Zero Variance)", "3 Features", "Constant Integer / String", "EmployeeCount, StandardHours, Over18 dropped entirely"),
        ("Removed (Unique ID)", "1 Feature", "Integer Sequence", "EmployeeNumber dropped to prevent memorization"),
        ("Total Raw Attributes", "35 Attributes", "Mixed Tabular Data", "Pristine hygiene: 0 missing values, 0 duplicate rows"),
        ("Transformed Feature Space", "51 Normalized Columns", "Float64 Dense Vector", "Standardized and orthogonal inputs ready for model training"),
    ]
    for idx, rvals in enumerate(sch_rows):
        r = sch_tab.rows[idx + 1]
        for col_idx, val in enumerate(rvals):
            r.cells[col_idx].paragraphs[0].add_run(val)
    style_table(sch_tab, [1.6, 1.3, 1.5, 2.4], header_bg="1F3A5F")

    doc.add_page_break()

    # -------------------------------------------------------------
    # CHAPTER 5: METHODOLOGY / ML MODEL DEVELOPMENT
    # -------------------------------------------------------------
    p_c5 = doc.add_paragraph()
    p_c5.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_c5.paragraph_format.space_before = Pt(10)
    p_c5.paragraph_format.space_after = Pt(4)
    rc5 = p_c5.add_run("CHAPTER 5\nMETHODOLOGY / ML MODEL DEVELOPMENT")
    rc5.font.name = "Times New Roman"
    rc5.font.size = Pt(16)
    rc5.font.bold = True

    doc.add_paragraph().add_run("5.1 PROBLEM FORMULATION & MATHEMATICAL FOUNDATION").font.bold = True
    doc.add_paragraph(
        "Employee attrition prediction is formulated as supervised binary classification. Given an employee feature vector x_i in R^51, "
        "the model predicts the probability P(y_i = 1 | x_i), where y_i = 1 denotes voluntary departure (Attrition = Yes) and y_i = 0 denotes retention. "
        "Because the minority class accounts for only 16.12% of records, conventional empirical risk minimization produces a majority-class bias. "
        "To correct this, we incorporate cost-sensitive loss weighting:"
    )
    doc.add_paragraph(
        "L_w(y, y_hat) = - [ w_1 * y * log(y_hat) + w_0 * (1 - y) * log(1 - y_hat) ]\n"
        "where w_1 = N / (2 * N_1) = 1470 / (2 * 237) = 3.101, and w_0 = N / (2 * N_0) = 1470 / (2 * 1233) = 0.596."
    )

    doc.add_paragraph().add_run("5.2 SUPERVISED LEARNING ALGORITHMS & INDUCTIVE BIASES").font.bold = True
    doc.add_paragraph(
        "Five distinct classification algorithms were developed to evaluate diverse mathematical inductive biases:\n"
        "1. Logistic Regression: Linear decision surface optimizing log-likelihood with liblinear solver and balanced weighting.\n"
        "2. Decision Tree Classifier: Recursive binary recursive partitioning based on Gini impurity reduction.\n"
        "3. Random Forest Classifier: Ensemble bagging of decorrelated decision trees with bootstrap aggregation.\n"
        "4. K-Nearest Neighbors (KNN): Non-parametric instance-based spatial metric clustering using Euclidean distance.\n"
        "5. Support Vector Machine (SVM): Maximum-margin hyperplane with non-linear Radial Basis Function (RBF) kernel:\n"
        "   K(x, x') = exp( - gamma * || x - x' ||^2 )"
    )

    doc.add_paragraph().add_run("5.3 OPTIMAL DECISION THRESHOLD TUNING (0.33)").font.bold = True
    doc.add_paragraph(
        "Standard machine learning classifiers apply a fixed threshold of 0.50. In imbalanced domains, this severely suppresses recall. "
        "By mapping the Precision-Recall curve under cross-validation, we optimized the threshold theta to maximize the harmonic mean (F1-score):\n"
        "theta* = argmax_theta [ 2 * (P(theta) * R(theta)) / (P(theta) + R(theta)) ]\n"
        "The empirical optimum was identified at theta = 0.33. Lowering the decision boundary from 0.50 to 0.33 increased test F1-score from "
        "0.539 to 0.5594, precision to 0.5556, and recall to 0.5634 (catching 40 out of 71 real leavers)."
    )

    doc.add_paragraph().add_run("5.4 DUAL-LAYER EXPLAINABILITY ENGINE").font.bold = True
    doc.add_paragraph(
        "• Global Explainability (Permutation Importance): Evaluates the drop in model F1-score when a feature column is randomly shuffled:\n"
        "  I(f) = F1_baseline - F1_shuffled(f)\n"
        "• Local Explainability (SHAP Shapley Values): Decomposes individual predictions into additive feature contributions:\n"
        "  f(x) = phi_0 + SUM_j [ phi_j(x) ]\n"
        "• 170x SHAP Acceleration: KernelExplainer evaluates predictions across subsets of background data. Running across 100 raw background rows "
        "required ~57 seconds. By summarizing the background dataset into 10 weighted centroids using k-means clustering (shap.kmeans) and setting "
        "nsamples=100, execution latency dropped to 0.32 seconds with zero degradation in top factor rankings."
    )

    doc.add_page_break()

    # -------------------------------------------------------------
    # CHAPTER 6: IMPLEMENTATION DETAILS
    # -------------------------------------------------------------
    p_c6 = doc.add_paragraph()
    p_c6.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_c6.paragraph_format.space_before = Pt(10)
    p_c6.paragraph_format.space_after = Pt(4)
    rc6 = p_c6.add_run("CHAPTER 6\nIMPLEMENTATION DETAILS")
    rc6.font.name = "Times New Roman"
    rc6.font.size = Pt(16)
    rc6.font.bold = True

    doc.add_paragraph().add_run("6.1 TECHNOLOGY STACK MATRIX").font.bold = True
    doc.add_paragraph("Table 6.1 details the software components and engineering libraries utilized in the project:")

    tech_tab = doc.add_table(rows=7, cols=3)
    tech_tab.alignment = WD_TABLE_ALIGNMENT.CENTER
    tech_headers = ["Layer / Subsystem", "Technology / Library", "Role & Engineering Responsibility"]
    for j, h in enumerate(tech_headers):
        tech_tab.rows[0].cells[j].paragraphs[0].add_run(h)

    tech_data = [
        ("Core Language", "Python 3.14.3", "Runtime environment, pipeline orchestration, and test execution"),
        ("Data Preprocessing", "Pandas 2.2, NumPy 2.0, Scipy 1.11", "Data loading, ColumnTransformer array manipulation, and statistics"),
        ("Machine Learning", "Scikit-Learn 1.5, Joblib 1.3", "Pipeline implementation, GridSearchCV, model training, and serialization"),
        ("Model Explainability", "SHAP 0.45", "Fast Shapley Additive Explanations with k-means background clustering"),
        ("Interactive Web App", "Streamlit 1.36, Plotly 5.20", "Glassmorphic responsive dashboard, persona presets, and batch CSV scoring"),
        ("Quality Assurance", "Python Unittest (Standard Library)", "Automated test suite verifying data, preprocessing, SHAP, and metrics"),
    ]
    for idx, rvals in enumerate(tech_data):
        r = tech_tab.rows[idx + 1]
        for col_idx, val in enumerate(rvals):
            r.cells[col_idx].paragraphs[0].add_run(val)
    style_table(tech_tab, [1.8, 2.0, 3.0], header_bg="1F3A5F")

    doc.add_paragraph().add_run("6.2 CORE MODULE IMPLEMENTATIONS").font.bold = True
    doc.add_paragraph(
        "• src/preprocessing.py: Implements load_data(), prepare_dataframe(), build_preprocessor(), and transformed_feature_names(). "
        "Separates 23 numerical features and 7 categorical features.\n"
        "• src/train.py: Manages baseline training, 5-fold cross-validation, GridSearchCV / RandomizedSearchCV tuning, champion model selection, "
        "and artifact persistence (best_model.pkl, model_results.json, and evaluation plots).\n"
        "• src/predict.py: Exposes predict_probability(), batch_predict(), and explain_prediction(). Employs _EXPLAINER_CACHE and shap.kmeans "
        "to deliver sub-second local XAI explanations.\n"
        "• app/app.py: Streamlit dashboard structured into Overview, Risk Assessment (Single & Batch CSV), Models, and Explainability pages."
    )

    doc.add_page_break()

    # -------------------------------------------------------------
    # CHAPTER 7: TESTING, RESULTS AND PERFORMANCE EVALUATION
    # -------------------------------------------------------------
    p_c7 = doc.add_paragraph()
    p_c7.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_c7.paragraph_format.space_before = Pt(10)
    p_c7.paragraph_format.space_after = Pt(4)
    rc7 = p_c7.add_run("CHAPTER 7\nTESTING, RESULTS AND PERFORMANCE EVALUATION")
    rc7.font.name = "Times New Roman"
    rc7.font.size = Pt(16)
    rc7.font.bold = True

    doc.add_paragraph().add_run("7.1 AUTOMATED UNIT TESTING (10/10 PASSING TESTS)").font.bold = True
    doc.add_paragraph(
        "A rigorous automated test suite was developed in tests/test_pipeline.py covering all critical functional modules. "
        "The test suite executed with a 100% pass rate in 4.88 seconds:\n"
        "• test_load_data_shape_and_columns: PASS (1,470 records, 35 attributes, 0 missing values)\n"
        "• test_prepare_dataframe: PASS (Binary target, feature columns match specification)\n"
        "• test_preprocessor_transformation: PASS (Transforms raw inputs into 51 normalized columns)\n"
        "• test_load_model_bundle_keys: PASS (Verifies pipeline, estimator, threshold, feature_names)\n"
        "• test_predict_probability_bounds: PASS (0.0 <= P <= 1.0)\n"
        "• test_risk_level_mapping: PASS (Correct mapping of Low, Medium, High, Critical tiers)\n"
        "• test_predict_with_explanation: PASS (SHAP explanation returns ranked feature attributions)\n"
        "• test_batch_predict: PASS (DataFrame multi-record batch scoring with risk tiers)\n"
        "• test_compute_metrics: PASS (Validates Accuracy, Precision, Recall, F1, ROC-AUC calculations)\n"
        "• test_compute_threshold_curve: PASS (Optimal threshold calculation from Precision-Recall curve)"
    )

    doc.add_paragraph().add_run("7.2 MODEL BENCHMARK & COMPARATIVE EVALUATION").font.bold = True
    doc.add_paragraph("Table 7.1 presents the complete performance comparison on the 30% hold-out test set (441 employees: 370 stayed, 71 departed):")

    res_table = doc.add_table(rows=6, cols=7)
    res_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    res_headers = ["Model Name", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "5-Fold CV F1"]
    for j, h in enumerate(res_headers):
        res_table.rows[0].cells[j].paragraphs[0].add_run(h)

    res_data = [
        ("Support Vector Machine (Champion)", "0.832", "0.474", "0.634", "0.539", "0.817", "0.553"),
        ("Logistic Regression", "0.757", "0.366", "0.676", "0.480", "0.816", "0.512"),
        ("Random Forest", "0.844", "0.488", "0.423", "0.455", "0.780", "0.514"),
        ("Decision Tree", "0.741", "0.320", "0.577", "0.414", "0.698", "0.408"),
        ("K-Nearest Neighbors", "0.841", "0.519", "0.197", "0.286", "0.611", "0.284"),
    ]
    for idx, rvals in enumerate(res_data):
        r = res_table.rows[idx + 1]
        for col_idx, val in enumerate(rvals):
            r.cells[col_idx].paragraphs[0].add_run(val)
    style_table(res_table, [1.8, 0.8, 0.8, 0.8, 0.8, 0.8, 1.0], header_bg="1F3A5F")

    add_figure_with_caption(doc, PLOTS_DIR / "model_comparison.png", "Figure 7.1: Comparative F1-Score Benchmark Across All 5 ML Classifiers")
    add_figure_with_caption(doc, PLOTS_DIR / "roc_curves.png", "Figure 7.2: ROC Curves Benchmark Across 5 Classifiers with AUC Values")

    doc.add_paragraph().add_run("7.3 HYPERPARAMETER TUNING & THRESHOLD OPTIMIZATION").font.bold = True
    doc.add_paragraph(
        "Hyperparameter tuning on 5-fold cross-validation confirmed optimal parameters for SVM: C = 0.5, gamma = 'scale', RBF kernel. "
        "Lowering the decision threshold to 0.33 maximized the F1-score to 0.5594, raising precision to 0.5556, recall to 0.5634, and overall accuracy to 0.8571."
    )

    add_figure_with_caption(doc, PLOTS_DIR / "confusion_matrix.png", "Figure 7.3: Confusion Matrix of Champion SVM Model at Threshold 0.33")

    doc.add_paragraph().add_run("7.4 CONFUSION MATRIX & ECONOMIC COST-BENEFIT MATRIX").font.bold = True
    doc.add_paragraph(
        "At threshold 0.33, the hold-out test set confusion matrix records:\n"
        "• True Negatives (TN): 338 retained employees correctly classified as Stay.\n"
        "• False Positives (FP): 32 retained employees flagged for proactive check-in (Cost: ~$1,000 each in manager time).\n"
        "• False Negatives (FN): 31 departing employees missed by model (Cost: ~$30,000–$75,000 replacement cost each).\n"
        "• True Positives (TP): 40 real leavers caught and flagged for retention intervention.\n"
        "Economic ROI: Successfully catching 40 leavers prevents an estimated $1.2M to $3.0M in replacement costs against an intervention budget of under $40,000."
    )

    doc.add_page_break()

    # -------------------------------------------------------------
    # CHAPTER 8: CONCLUSION AND FUTURE SCOPE
    # -------------------------------------------------------------
    p_c8 = doc.add_paragraph()
    p_c8.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_c8.paragraph_format.space_before = Pt(10)
    p_c8.paragraph_format.space_after = Pt(4)
    rc8 = p_c8.add_run("CHAPTER 8\nCONCLUSION AND FUTURE SCOPE")
    rc8.font.name = "Times New Roman"
    rc8.font.size = Pt(16)
    rc8.font.bold = True

    doc.add_paragraph().add_run("8.1 CONCLUSION").font.bold = True
    doc.add_paragraph(
        "This project has successfully designed, implemented, and validated an explainable, end-to-end Machine Learning system for employee attrition "
        "prediction that directly overcomes the fundamental challenges of real-world class imbalance and black-box decision making. "
        "By enforcing a leak-free scikit-learn Pipeline, applying balanced loss weighting, and optimizing decision thresholds to 0.33, the champion Support "
        "Vector Machine model achieved an F1-score of 0.5594 and ROC-AUC of 0.8168 on unseen hold-out data. "
        "Furthermore, by accelerating local SHAP explainability from 57 seconds to 0.32 seconds, the platform empowers enterprise human resource leadership "
        "with instant, transparent diagnostic intelligence."
    )

    doc.add_paragraph().add_run("8.2 FUTURE SCOPE").font.bold = True
    doc.add_paragraph(
        "• Enterprise HRIS API Integration: Connect directly to enterprise HR systems such as Workday, SAP SuccessFactors, and BambooHR via secure REST webhooks.\n"
        "• Natural Language Processing (NLP): Ingest free-text annual review comments and pulse surveys using transformer embeddings (BERT/RoBERTa) to capture subtle sentiment shifts.\n"
        "• Deep Survival Analysis: Transition from binary classification to Cox Proportional Hazards survival curves predicting the expected month of departure.\n"
        "• Algorithmic Fairness Audits: Continuously monitor demographic parity across gender, age, and departmental subgroups to ensure non-discriminatory retention practices."
    )

    doc.add_page_break()

    # -------------------------------------------------------------
    # REFERENCES
    # -------------------------------------------------------------
    p_ref = doc.add_paragraph()
    p_ref.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_ref.paragraph_format.space_before = Pt(10)
    p_ref.paragraph_format.space_after = Pt(14)
    rref = p_ref.add_run("REFERENCES")
    rref.font.name = "Times New Roman"
    rref.font.size = Pt(16)
    rref.font.bold = True

    refs = [
        "1. S. M. Lundberg and S. I. Lee, “A unified approach to interpreting model predictions,” in Advances in Neural Information Processing Systems (NeurIPS 2017), vol. 30, pp. 4765–4774, 2017.",
        "2. F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, et al., “Scikit-learn: Machine learning in Python,” Journal of Machine Learning Research, vol. 12, pp. 2825–2830, Nov. 2011.",
        "3. IBM Watson Analytics, “HR Employee Attrition and Performance Dataset,” IBM Corporation, Kaggle Open Datasets, 2016. [Online]. Available: https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset.",
        "4. N. V. Chawla, K. W. Bowyer, L. O. Hall, and W. P. Kegelmeyer, “SMOTE: synthetic minority over-sampling technique,” Journal of Artificial Intelligence Research, vol. 16, pp. 321–357, Jun. 2002.",
        "5. L. Breiman, “Random forests,” Machine Learning, vol. 45, no. 1, pp. 5–32, Oct. 2001. DOI: 10.1023/A:1010933404324.",
        "6. C. Cortes and V. Vapnik, “Support-vector networks,” Machine Learning, vol. 20, no. 3, pp. 273–297, Sep. 1995. DOI: 10.1007/BF00994018.",
        "7. T. Hastie, R. Tibshirani, and J. Friedman, The Elements of Statistical Learning: Data Mining, Inference, and Prediction, 2nd ed. New York: Springer, 2009.",
        "8. J. Davis and M. Goadrich, “The relationship between Precision-Recall and ROC curves,” in Proc. 23rd International Conference on Machine Learning (ICML), 2006, pp. 233–240.",
        "9. H. He and E. A. Garcia, “Learning from imbalanced data,” IEEE Transactions on Knowledge and Data Engineering, vol. 21, no. 9, pp. 1263–1284, Sep. 2009.",
        "10. Society for Human Resource Management (SHRM), “Human Capital Benchmarking Report,” SHRM Research, Alexandria, VA, USA, Tech. Rep., 2023.",
    ]
    for r in refs:
        p = doc.add_paragraph(r)
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.line_spacing = 1.15
        p.paragraph_format.space_after = Pt(6)

    doc.add_page_break()

    # -------------------------------------------------------------
    # APPENDICES
    # -------------------------------------------------------------
    p_ap = doc.add_paragraph()
    p_ap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_ap.paragraph_format.space_before = Pt(10)
    p_ap.paragraph_format.space_after = Pt(8)
    rap = p_ap.add_run("APPENDIX A\nSYSTEM SNAPSHOTS AND USER INTERFACE")
    rap.font.name = "Times New Roman"
    rap.font.size = Pt(16)
    rap.font.bold = True

    add_figure_with_caption(doc, EDA_DIR / "eda_target_distribution.png", "Snapshot 1: Workforce Class Imbalance Visualization (83.9% vs. 16.1%)")
    add_figure_with_caption(doc, EDA_DIR / "eda_overtime_marital.png", "Snapshot 2: OverTime & Marital Status Attrition Disparity Analysis")
    add_figure_with_caption(doc, EDA_DIR / "eda_satisfaction_trends.png", "Snapshot 3: Monotonic Churn Trends Across Employee Satisfaction Dimensions")
    add_figure_with_caption(doc, PLOTS_DIR / "confusion_matrix.png", "Snapshot 4: Confusion Matrix Heatmap Displaying Calibrated Class Counts")
    add_figure_with_caption(doc, PLOTS_DIR / "feature_importance.png", "Snapshot 5: Global Permutation Importance Ranking of Churn Drivers")

    doc.add_page_break()

    p_apb = doc.add_paragraph()
    p_apb.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_apb.paragraph_format.space_before = Pt(10)
    p_apb.paragraph_format.space_after = Pt(8)
    rapb = p_apb.add_run("APPENDIX B\nSOURCE CODE DIRECTORY STRUCTURE")
    rapb.font.name = "Times New Roman"
    rapb.font.size = Pt(16)
    rapb.font.bold = True

    doc.add_paragraph(
        "Employee-Attrition-ML/\n"
        "├── data/\n"
        "│   ├── employee_attrition.csv            # IBM HR Analytics dataset (1,470 records)\n"
        "│   └── DATA_DICTIONARY.md                # Comprehensive 35-variable data dictionary\n"
        "├── notebooks/\n"
        "│   └── exploratory_analysis.ipynb        # Fully executed EDA notebook with embedded plots\n"
        "├── src/\n"
        "│   ├── preprocessing.py                  # Data loading, column definitions, ColumnTransformer\n"
        "│   ├── train.py                          # 5-model training, hyperparameter tuning & artifacts\n"
        "│   ├── evaluate.py                       # Evaluation metrics, confusion matrix & ROC curves\n"
        "│   └── predict.py                        # Probability inference, batch scoring & fast SHAP\n"
        "├── models/\n"
        "│   └── best_model.pkl                    # Serialized pipeline bundle (model + preprocessor + explainer)\n"
        "├── outputs/\n"
        "│   ├── model_results.json                # Benchmark metrics, ROC coordinates & confusion matrices\n"
        "│   └── plots/                            # confusion_matrix, roc_curves, model_comparison, feature_importance\n"
        "│       └── eda/                          # 6 high-resolution publication-grade EDA figures\n"
        "├── app/\n"
        "│   └── app.py                            # Streamlit dashboard (Overview, Predict, Models, Explainability)\n"
        "├── reports/\n"
        "│   ├── Final_Project_Report.docx         # Complete VTU format report adhering to Zero Trust standard\n"
        "│   ├── Final_Project_Report.md           # Markdown thesis documentation\n"
        "│   ├── Phase_1_Report.docx               # Phase 1 evaluation report (40 marks)\n"
        "│   └── Phase_1_Presentation.pptx         # Phase 1 evaluation presentation deck\n"
        "├── scripts/\n"
        "│   ├── generate_reports.py               # Automated report generation script\n"
        "│   ├── generate_presentation.py          # PowerPoint deck generation script\n"
        "│   ├── generate_eda_plots.py             # High-resolution EDA figures generator\n"
        "│   └── execute_notebook.py               # Headless EDA notebook execution engine\n"
        "├── tests/\n"
        "│   └── test_pipeline.py                  # Automated unit test suite (10/10 passing tests)\n"
        "├── Employee_Attrition_Presentation.pptx   # 25-slide final viva presentation deck\n"
        "├── VIVA_PREPARATION.md                   # Comprehensive viva defense & technical interview guide\n"
        "├── requirements.txt                      # Project dependencies\n"
        "└── README.md"
    )

    doc.add_page_break()

    p_apc = doc.add_paragraph()
    p_apc.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_apc.paragraph_format.space_before = Pt(10)
    p_apc.paragraph_format.space_after = Pt(8)
    rapc = p_apc.add_run("APPENDIX C\nGUIDE APPROVAL & CIE EVALUATION MARKS SHEET")
    rapc.font.name = "Times New Roman"
    rapc.font.size = Pt(16)
    rapc.font.bold = True

    eval_tab = doc.add_table(rows=7, cols=3)
    eval_tab.alignment = WD_TABLE_ALIGNMENT.CENTER
    eval_headers = ["Phase & Evaluation Component", "Maximum Marks", "Marks Awarded by Guide"]
    for j, h in enumerate(eval_headers):
        eval_tab.rows[0].cells[j].paragraphs[0].add_run(h)

    eval_data = [
        ("Phase 1: Problem Definition, Dataset, EDA & Preprocessing", "40 Marks", "_____ / 40 Marks"),
        ("Phase 2: Model Comparison, Tuning, Evaluation & XAI", "30 Marks", "_____ / 30 Marks"),
        ("Phase 2: Web Deployment & Automated Testing", "15 Marks", "_____ / 15 Marks"),
        ("Phase 2: Project Report & Documentation", "15 Marks", "_____ / 15 Marks"),
        ("Total Score (Out of 100 Marks)", "100 Marks", "_____ / 100 Marks"),
        ("Final Continuous Internal Evaluation (Downscaled to 10 Marks)", "10 Marks", "_____ / 10 Marks"),
    ]
    for idx, rvals in enumerate(eval_data):
        r = eval_tab.rows[idx + 1]
        for col_idx, val in enumerate(rvals):
            r.cells[col_idx].paragraphs[0].add_run(val)
    style_table(eval_tab, [3.2, 1.5, 1.8], header_bg="1F3A5F")

    doc.add_paragraph("\n\n")
    p_eval_sig = doc.add_paragraph()
    p_eval_sig.add_run("Guide Signature: ___________________________          Date: _____________________\n")
    p_eval_sig.add_run("HOD Signature with Seal: ___________________          Date: _____________________")

    # Save to both Final_Project_Report.docx and Employee_Attrition_VTU_Project_Report.docx
    out_path1 = REPORTS_DIR / "Final_Project_Report.docx"
    out_path2 = REPORTS_DIR / "Employee_Attrition_VTU_Project_Report.docx"
    doc.save(out_path1)
    doc.save(out_path2)
    print(f"Successfully generated:\n- {out_path1}\n- {out_path2}")


if __name__ == "__main__":
    build_report()
