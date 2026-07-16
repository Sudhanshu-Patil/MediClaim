"""Generate a sample insurance-policy PDF for testing the ingestion pipeline.

The document deliberately exercises every Phase-1 feature:
  * numbered sections + prose  → parent/child sentence-window chunking
  * "see Section X.X" phrases  → static REFERENCES edge detection
  * a 45-row fixed-column reimbursement table that spans multiple pages
    (repeated header row) → TableFormer + multi-page table merging +
    atomic table chunk + summary chunk

Usage (from repo root):
    python scripts/generate_sample_pdf.py                 # v1 → sample_docs/sample_policy.pdf
    python scripts/generate_sample_pdf.py --version 2     # revised copy (tests supersede)
    python scripts/generate_sample_pdf.py --out other.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

PROCEDURES = [
    ("OP-1001", "General practitioner consultation", "45.00", "10%", "No"),
    ("OP-1002", "Specialist consultation", "90.00", "15%", "No"),
    ("OP-1003", "Telehealth consultation", "35.00", "10%", "No"),
    ("OP-2001", "Basic metabolic panel", "28.50", "0%", "No"),
    ("OP-2002", "Complete blood count", "22.00", "0%", "No"),
    ("OP-2003", "Lipid panel", "31.00", "0%", "No"),
    ("OP-2004", "HbA1c test", "26.00", "0%", "No"),
    ("OP-2005", "Thyroid function panel", "48.00", "0%", "No"),
    ("OP-3001", "Chest X-ray (2 views)", "110.00", "20%", "No"),
    ("OP-3002", "Abdominal ultrasound", "165.00", "20%", "No"),
    ("OP-3003", "MRI brain without contrast", "620.00", "20%", "Yes"),
    ("OP-3004", "MRI lumbar spine", "580.00", "20%", "Yes"),
    ("OP-3005", "CT chest with contrast", "450.00", "20%", "Yes"),
    ("OP-3006", "Mammogram (screening)", "135.00", "0%", "No"),
    ("OP-3007", "Bone density scan (DEXA)", "125.00", "10%", "No"),
    ("OP-4001", "Physical therapy session", "75.00", "15%", "No"),
    ("OP-4002", "Occupational therapy session", "78.00", "15%", "No"),
    ("OP-4003", "Speech therapy session", "82.00", "15%", "Yes"),
    ("OP-4004", "Chiropractic adjustment", "55.00", "25%", "No"),
    ("OP-4005", "Acupuncture session", "60.00", "25%", "Yes"),
    ("OP-5001", "Minor wound suturing", "140.00", "10%", "No"),
    ("OP-5002", "Skin lesion excision", "210.00", "15%", "Yes"),
    ("OP-5003", "Joint injection (corticosteroid)", "160.00", "15%", "No"),
    ("OP-5004", "Colonoscopy (screening)", "720.00", "0%", "Yes"),
    ("OP-5005", "Upper GI endoscopy", "680.00", "15%", "Yes"),
    ("OP-5006", "Cataract surgery (outpatient)", "1850.00", "20%", "Yes"),
    ("OP-6001", "Influenza vaccination", "18.00", "0%", "No"),
    ("OP-6002", "Pneumococcal vaccination", "42.00", "0%", "No"),
    ("OP-6003", "Shingles vaccination", "95.00", "0%", "No"),
    ("OP-6004", "Travel vaccination package", "160.00", "50%", "No"),
    ("OP-7001", "Diabetic retinopathy screening", "88.00", "0%", "No"),
    ("OP-7002", "Glaucoma screening", "72.00", "10%", "No"),
    ("OP-7003", "Audiometry assessment", "64.00", "10%", "No"),
    ("OP-7004", "Spirometry / lung function", "58.00", "10%", "No"),
    ("OP-8001", "Mental health counselling (45 min)", "95.00", "15%", "No"),
    ("OP-8002", "Psychiatric evaluation", "180.00", "15%", "Yes"),
    ("OP-8003", "Group therapy session", "40.00", "10%", "No"),
    ("OP-9001", "Dietitian consultation", "70.00", "20%", "No"),
    ("OP-9002", "Smoking cessation program", "120.00", "0%", "No"),
    ("OP-9003", "Cardiac rehabilitation session", "105.00", "10%", "Yes"),
    ("OP-9004", "Sleep study (home-based)", "310.00", "20%", "Yes"),
    ("OP-9005", "Allergy panel testing", "150.00", "15%", "No"),
    ("OP-9006", "Wart / verruca removal", "85.00", "25%", "No"),
    ("OP-9007", "Ear irrigation", "38.00", "10%", "No"),
    ("OP-9008", "Holter monitor (24h)", "190.00", "20%", "No"),
]


def build_pdf(out_path: Path, version: int) -> None:
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(
        str(out_path), pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2 * cm, bottomMargin=2 * cm,
        title="Acme Health Outpatient Claims Policy",
    )
    story = []
    p = lambda text, style="BodyText": story.append(Paragraph(text, styles[style]))

    p("Acme Health — Outpatient Claims Adjudication Policy", "Title")
    p(f"Policy document version {version}. For internal adjudication use.", "Italic")
    story.append(Spacer(1, 12))

    p("1. Purpose and Scope", "Heading1")
    p(
        "This policy defines the rules under which outpatient medical claims are "
        "adjudicated for Acme Health members. It applies to all outpatient services "
        "rendered by in-network and out-of-network providers. Claims for inpatient "
        "admissions are governed by a separate policy. Adjudicators must verify "
        "member eligibility on the date of service before applying the rules in "
        "this document. Where a service requires prior authorization, the "
        "authorization reference must be attached to the claim; the full list of "
        "such services is set out in Section 3.1."
    )
    p(
        "Any claim that cannot be matched to a procedure code listed in Section 3.1 "
        "must be routed to manual review. Exclusions are enumerated in Section 2.2, "
        "and the appeals process is described in Section 4."
    )

    p("2. Coverage Rules", "Heading1")
    p("2.1 General Coverage Conditions", "Heading2")
    p(
        "Outpatient services are covered when they are medically necessary, "
        "rendered by a licensed provider acting within the scope of their license, "
        "and billed with a procedure code listed in the reimbursement schedule. "
        "Medical necessity is presumed for preventive services with a 0% copayment "
        "in Section 3.1. For all other services, the diagnosis code on the claim "
        "must support the billed procedure. Claims submitted more than 180 days "
        "after the date of service are denied for untimely filing unless the "
        "member can demonstrate good cause, as described in Section 4."
        + (
            " Effective with this revision, telehealth consultations are covered "
            "at parity with in-person visits."
            if version >= 2
            else ""
        )
    )
    p("2.2 Exclusions", "Heading2")
    p(
        "The following are excluded from outpatient coverage: cosmetic procedures "
        "without reconstructive indication; experimental or investigational "
        "treatments; services rendered outside the coverage period; charges "
        "exceeding the maximum benefit amounts listed in Section 3.1; and "
        "convenience items. Denials based on this subsection may be appealed "
        "under Section 4."
    )

    p("3. Reimbursement", "Heading1")
    p("3.1 Reimbursement Schedule", "Heading2")
    p(
        "The schedule below lists the maximum benefit per procedure, the member "
        "copayment percentage, and whether prior authorization is required. "
        "Amounts are in USD and apply per occurrence."
    )

    header = ["Procedure Code", "Description", "Max Benefit (USD)", "Copay", "Prior Auth"]
    rows = [list(r) for r in PROCEDURES]
    if version >= 2:
        rows[10][2] = "680.00"   # MRI brain: raised benefit in v2
        rows[10][4] = "No"
        rows.append(["OP-9009", "Continuous glucose monitor fitting", "145.00", "10%", "No"])
    table = Table([header] + rows, repeatRows=1, colWidths=[2.8 * cm, 6.4 * cm, 3.0 * cm, 1.8 * cm, 2.2 * cm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2f4f6f")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef2f6")]),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 12))

    p("3.2 Coordination of Benefits", "Heading2")
    p(
        "Where a member holds concurrent coverage with another insurer, Acme Health "
        "pays secondary to the primary plan. The combined reimbursement must not "
        "exceed the maximum benefit listed in Section 3.1 for the billed procedure. "
        "Adjudicators must request the primary plan's explanation of benefits "
        "before finalizing any coordinated claim."
    )

    story.append(PageBreak())
    p("4. Appeals", "Heading1")
    p(
        "A member or provider may appeal an adverse determination within 60 days of "
        "the denial notice. First-level appeals are reviewed by an adjudicator not "
        "involved in the original decision. Appeals of medical-necessity denials "
        "require review by a licensed clinician. Denials for untimely filing, per "
        "Section 2.1, are upheld unless documented good cause exists. Coverage "
        "questions arising under Section 2.2 exclusions follow the same process. "
        "Unresolved second-level appeals proceed to external review as defined in "
        "Appendix A."
    )

    p("Appendix A — External Review", "Heading1")
    p(
        "External review is conducted by an independent review organization. The "
        "organization's determination is binding on Acme Health. Members retain "
        "any rights available under applicable law. Requests must include the "
        "final internal denial letter and supporting clinical records, as "
        "described in Section 4."
    )

    doc.build(story)
    print(f"Wrote {out_path} (version {version}, {len(rows)} table rows)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", type=int, default=1, help="1 = original, 2 = revised (tests supersede)")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    out = Path(args.out) if args.out else Path("sample_docs") / "sample_policy.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    build_pdf(out, args.version)


if __name__ == "__main__":
    main()
