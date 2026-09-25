DO $$ BEGIN
  IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname='voyage4_writer') THEN
    CREATE ROLE voyage4_writer NOLOGIN;
  END IF;
END $$;
GRANT voyage4_writer TO arxiv;
GRANT CONNECT ON DATABASE arxiv TO voyage4_writer;
GRANT USAGE ON SCHEMA public TO voyage4_writer;
GRANT SELECT ON public.papers_paper TO voyage4_writer;
ALTER SCHEMA voyage4 OWNER TO voyage4_writer;
ALTER TABLE voyage4.embeddings OWNER TO voyage4_writer;
ALTER TABLE voyage4.rolling30 OWNER TO voyage4_writer;
ALTER TABLE voyage4.control OWNER TO voyage4_writer;
ALTER TABLE voyage4.pending_papers OWNER TO voyage4_writer;
