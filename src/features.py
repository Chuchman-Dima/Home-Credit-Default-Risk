"""
features.py - Feature engineering for Home Credit Default Risk.

Joins all 7 tables on SK_ID_CURR and creates aggregated features.
Читає з локального parquet кешу (швидко). Fallback на PostgreSQL.
"""

import pandas as pd
import numpy as np
from src.db import load_local


# ------------------------------------------------------------------ #
#  Individual table feature builders                                  #
# ------------------------------------------------------------------ #

def build_bureau_features(bureau: pd.DataFrame) -> pd.DataFrame:
    grp = bureau.groupby("sk_id_curr")
    agg = grp.agg(
        bureau_loan_count       =("sk_id_bureau",           "count"),
        bureau_active_loans     =("credit_active",          lambda x: (x == "Active").sum()),
        bureau_closed_loans     =("credit_active",          lambda x: (x == "Closed").sum()),
        bureau_avg_days_credit  =("days_credit",            "mean"),
        bureau_avg_days_enddate =("days_credit_enddate",    "mean"),
        bureau_total_debt       =("amt_credit_sum_debt",    "sum"),
        bureau_total_credit     =("amt_credit_sum",         "sum"),
        bureau_avg_overdue      =("amt_credit_sum_overdue", "mean"),
        bureau_max_overdue      =("credit_day_overdue",     "max"),
        bureau_sum_overdue      =("credit_day_overdue",     "sum"),
    ).reset_index()
    agg["bureau_debt_ratio"] = agg["bureau_total_debt"] / (agg["bureau_total_credit"] + 1)
    return agg


def build_bureau_balance_features(
    bureau: pd.DataFrame,
    bureau_balance: pd.DataFrame,
) -> pd.DataFrame:
    bb = bureau_balance.merge(
        bureau[["sk_id_bureau", "sk_id_curr"]], on="sk_id_bureau", how="left"
    )
    grp = bb.groupby("sk_id_curr")
    agg = grp.agg(
        bb_months_count       =("months_balance", "count"),
        bb_avg_months_balance =("months_balance", "mean"),
        bb_dpd_count          =("status",         lambda x: (x == "C").sum()),
        bb_overdue_count      =("status",         lambda x: x.isin(["1","2","3","4","5"]).sum()),
    ).reset_index()
    return agg


def build_prev_app_features(prev: pd.DataFrame) -> pd.DataFrame:
    grp = prev.groupby("sk_id_curr")
    agg = grp.agg(
        prev_app_count          =("sk_id_prev",           "count"),
        prev_app_approved       =("name_contract_status", lambda x: (x == "Approved").sum()),
        prev_app_refused        =("name_contract_status", lambda x: (x == "Refused").sum()),
        prev_avg_credit         =("amt_credit",           "mean"),
        prev_avg_annuity        =("amt_annuity",          "mean"),
        prev_avg_application    =("amt_application",      "mean"),
        prev_avg_down_payment   =("amt_down_payment",     "mean"),
        prev_max_credit         =("amt_credit",           "max"),
        prev_avg_days_decision  =("days_decision",        "mean"),
        prev_avg_days_last_due  =("days_last_due",        "mean"),
    ).reset_index()
    agg["prev_approval_rate"] = agg["prev_app_approved"] / (agg["prev_app_count"] + 1)
    return agg


def build_pos_cash_features(pos: pd.DataFrame) -> pd.DataFrame:
    grp = pos.groupby("sk_id_curr")
    agg = grp.agg(
        pos_months_count  =("months_balance",       "count"),
        pos_avg_instalment=("cnt_instalment",        "mean"),
        pos_avg_future    =("cnt_instalment_future", "mean"),
        pos_completed     =("name_contract_status",  lambda x: (x == "Completed").sum()),
        pos_active        =("name_contract_status",  lambda x: (x == "Active").sum()),
        pos_max_dpd       =("sk_dpd",                "max"),
        pos_avg_dpd       =("sk_dpd",                "mean"),
        pos_max_dpd_def   =("sk_dpd_def",            "max"),
    ).reset_index()
    return agg


def build_installments_features(inst: pd.DataFrame) -> pd.DataFrame:
    inst = inst.copy()
    inst["payment_diff"] = inst["amt_instalment"] - inst["amt_payment"]
    inst["days_late"]    = (inst["days_entry_payment"] - inst["days_instalment"]).clip(lower=0)

    grp = inst.groupby("sk_id_curr")
    agg = grp.agg(
        inst_count            =("amt_instalment", "count"),
        inst_avg_amount       =("amt_instalment", "mean"),
        inst_total_payment    =("amt_payment",    "sum"),
        inst_avg_payment_diff =("payment_diff",   "mean"),
        inst_max_payment_diff =("payment_diff",   "max"),
        inst_avg_days_late    =("days_late",      "mean"),
        inst_max_days_late    =("days_late",      "max"),
        inst_late_count       =("days_late",      lambda x: (x > 0).sum()),
    ).reset_index()
    agg["inst_late_rate"] = agg["inst_late_count"] / (agg["inst_count"] + 1)
    return agg


def build_credit_card_features(cc: pd.DataFrame) -> pd.DataFrame:
    grp = cc.groupby("sk_id_curr")
    agg = grp.agg(
        cc_months_count     =("months_balance",           "count"),
        cc_avg_balance      =("amt_balance",              "mean"),
        cc_max_balance      =("amt_balance",              "max"),
        cc_avg_credit_limit =("amt_credit_limit_actual",  "mean"),
        cc_avg_drawings     =("amt_drawings_current",     "mean"),
        cc_avg_payment      =("amt_payment_current",      "mean"),
        cc_avg_receivable   =("amt_receivable_principal", "mean"),
        cc_max_dpd          =("sk_dpd",                   "max"),
        cc_avg_dpd          =("sk_dpd",                   "mean"),
    ).reset_index()
    agg["cc_utilization"] = agg["cc_avg_balance"] / (agg["cc_avg_credit_limit"] + 1)
    return agg


# ------------------------------------------------------------------ #
#  Main builder                                                        #
# ------------------------------------------------------------------ #

def build_features(app: pd.DataFrame = None, use_local: bool = True) -> pd.DataFrame:
    """
    Build full feature matrix by joining all 7 tables.

    Args:
        app:       application_train DataFrame (завантажується якщо None)
        use_local: True  → читати з parquet (швидко, ~5-15с)
                   False → читати з PostgreSQL (повільно, ~3-10хв)

    Returns:
        DataFrame з усіма ознаками, колонка TARGET збережена
    """
    loader = load_local if use_local else __import__("src.db", fromlist=["load_table"]).load_table

    print("Завантаження таблиць...")

    if app is None:
        app = loader("application_train")
    app.columns = [c.lower() for c in app.columns]

    bureau       = loader("bureau")
    bureau_bal   = loader("bureau_balance")
    prev_app     = loader("previous_application")
    pos_cash     = loader("pos_cash_balance")
    installments = loader("installments_payments")
    credit_card  = loader("credit_card_balance")

    for df in [bureau, bureau_bal, prev_app, pos_cash, installments, credit_card]:
        df.columns = [c.lower() for c in df.columns]

    print("\nПобудова ознак...")

    feat_bureau = build_bureau_features(bureau)
    feat_bb     = build_bureau_balance_features(bureau, bureau_bal)
    feat_prev   = build_prev_app_features(prev_app)
    feat_pos    = build_pos_cash_features(pos_cash)
    feat_inst   = build_installments_features(installments)
    feat_cc     = build_credit_card_features(credit_card)

    result = app.copy()
    for feat_df, name in [
        (feat_bureau, "bureau"),
        (feat_bb,     "bureau_balance"),
        (feat_prev,   "previous_application"),
        (feat_pos,    "pos_cash"),
        (feat_inst,   "installments"),
        (feat_cc,     "credit_card"),
    ]:
        result = result.merge(feat_df, on="sk_id_curr", how="left")
        print(f"  + {name:<25} {feat_df.shape[1]-1} ознак")

    print(f"\nМатриця ознак: {result.shape[0]:,} рядків × {result.shape[1]} колонок")
    return result


if __name__ == "__main__":
    df = build_features()
    print(df.head(2).to_string())