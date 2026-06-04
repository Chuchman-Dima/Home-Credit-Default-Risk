CREATE TABLE pos_cash_balance (
    SK_ID_PREV              INTEGER,
    SK_ID_CURR              INTEGER,
    MONTHS_BALANCE          INTEGER,
    CNT_INSTALMENT          NUMERIC,
    CNT_INSTALMENT_FUTURE   NUMERIC,
    NAME_CONTRACT_STATUS    VARCHAR(40),
    SK_DPD                  INTEGER,
    SK_DPD_DEF              INTEGER
);

COPY public.pos_cash_balance FROM 'C:/Home Credit Default Risk/Home-Credit-Default-Risk/data/POS_CASH_balance.csv' 
WITH (FORMAT csv, HEADER true, DELIMITER ',', ENCODING 'UTF8');