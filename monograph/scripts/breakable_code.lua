-- Let long inline code spans (file paths, dotted identifiers) break across lines.
-- pandoc renders inline code as \texttt{...}, which never breaks, so a long path like
-- data/eval/heteromini/eval_quality_four_seed_summary.json overruns the right margin.
--
-- We split a long span at path separators (/ _ . -) into several inline Code pieces with
-- an \allowbreak between them. Keeping the pieces as pandoc Code elements means their
-- LaTeX escaping is still handled by pandoc; only the \allowbreak is raw.

local MIN = 24   -- leave short spans (k=8, WS=k, R=6, .pt) untouched

function Code(el)
  if (utf8.len(el.text) or #el.text) < MIN then return nil end

  local out, buf = {}, ""
  for _, c in utf8.codes(el.text) do
    local ch = utf8.char(c)
    buf = buf .. ch
    if ch == "/" or ch == "_" or ch == "." or ch == "-" then
      out[#out + 1] = pandoc.Code(buf)
      out[#out + 1] = pandoc.RawInline("latex", "\\allowbreak{}")
      buf = ""
    end
  end
  if #buf > 0 then out[#out + 1] = pandoc.Code(buf) end
  return out
end
