CREATE TABLE installments_payments (
    SK_ID_PREV              INTEGER,
    SK_ID_CURR              INTEGER,
    NUM_STALMENT_VERSION    NUMERIC,
    NUM_INSTALMENT_NUMBER   INTEGER,
    DAYS_INSTALMENT         NUMERIC,
    DAYS_ENTRY_PAYMENT      NUMERIC,
    AMT_INSTALMENT          NUMERIC(15, 2),
    AMT_PAYMENT             NUMERIC(15, 2)
);

COPY public.installments_payments FROM 'C:/Home Credit Default Risk/Home-Credit-Default-Risk/data/installments_payments.csv' 
WITH (FORMAT csv, HEADER true, DELIMITER ',', ENCODING 'UTF8');