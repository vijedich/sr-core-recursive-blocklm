-- Give every table content-proportional column widths so the LaTeX writer emits
-- wrapping p{} columns instead of overrunning the margin. GFM tables carry no width
-- hints, so pandoc otherwise sets non-wrapping columns and wide tables overflow.
--
-- Narrow tables keep their natural (sub-full) width; only tables wider than the text
-- block are stretched to full width and forced to wrap.

local AVAIL = 86  -- approx. characters that fit across the text block in the body font

function Table(tbl)
  local ncol = #tbl.colspecs
  if ncol == 0 then return tbl end

  local maxlen = {}
  for i = 1, ncol do maxlen[i] = 1 end

  local function scan(rows)
    for _, row in ipairs(rows) do
      for i, cell in ipairs(row.cells) do
        if i <= ncol then
          local s = pandoc.utils.stringify(cell)
          local n = utf8.len(s) or #s
          if n > maxlen[i] then maxlen[i] = n end
        end
      end
    end
  end

  scan(tbl.head.rows)
  for _, body in ipairs(tbl.bodies) do scan(body.body) end

  -- Weight = content length + a flat per-column budget. The flat term lifts very narrow
  -- columns so a header/word doesn't butt against the next column, without letting a wide
  -- text column starve the rest.
  local PAD = 5
  local weight, total = {}, 0
  for i = 1, ncol do
    weight[i] = maxlen[i] + PAD
    total = total + weight[i]
  end
  local scale = (total > AVAIL) and 0.98 or (total / AVAIL)

  for i = 1, ncol do
    tbl.colspecs[i][2] = (weight[i] / total) * scale
  end
  return tbl
end
