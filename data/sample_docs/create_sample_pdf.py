"""
Run this once to generate a sample enterprise HR policy PDF for testing.
Uses only fpdf2 — install it first: pip install fpdf2
"""
from fpdf import FPDF

class EnterprisePDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 12)
        self.cell(0, 10, "ACME Corp - Confidential HR Policy Document", align="C")
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")


pdf = EnterprisePDF()
pdf.add_page()
pdf.set_font("Helvetica", size=11)

sections = [
    ("1. Leave Policy",
     "All full-time employees are entitled to 18 days of paid annual leave per calendar year. "
     "Leave must be approved by the direct manager at least 5 business days in advance. "
     "Unused leave can be carried forward up to a maximum of 5 days to the next calendar year."),

    ("2. Remote Work Policy",
     "Employees may work remotely up to 3 days per week subject to manager approval. "
     "Remote work is not permitted during the probation period of 6 months. "
     "All remote employees must be reachable during core hours of 10:00 AM to 4:00 PM IST."),

    ("3. Employee Personal Details (PII Test Section)",
     "The following employees are registered under this policy:\n"
     "- John Smith, Employee ID: EMP-1042, Email: john.smith@acmecorp.com, Phone: +91-9876543210\n"
     "- Sarah Johnson, Employee ID: EMP-2031, Email: sarah.j@acmecorp.com, Phone: +91-9845001234\n"
     "- Raj Patel, Employee ID: EMP-3017, Email: raj.patel@acmecorp.com, SSN: 123-45-6789"),

    ("4. Performance Review Process",
     "Performance reviews are conducted bi-annually in June and December. "
     "Employees are rated on a 5-point scale across four dimensions: "
     "technical skills, communication, collaboration, and delivery. "
     "A rating below 2 for two consecutive cycles may result in a Performance Improvement Plan (PIP)."),

    ("5. Grievance Redressal",
     "Employees can raise grievances by contacting HR at hr@acmecorp.com or calling +91-8000123456. "
     "All grievances are acknowledged within 2 business days and resolved within 15 business days. "
     "Employees may escalate unresolved grievances to the Chief People Officer at cpo@acmecorp.com."),
]

for title, body in sections:
    # Section heading
    pdf.set_font("Helvetica", "B", 11)
    pdf.multi_cell(0, 8, title)
    pdf.ln(1)
    # Section body
    pdf.set_font("Helvetica", size=10)
    pdf.multi_cell(0, 7, body)
    pdf.ln(5)

pdf.output("data/sample_docs/hr_policy.pdf")
print("✅ Created: data/sample_docs/hr_policy.pdf")