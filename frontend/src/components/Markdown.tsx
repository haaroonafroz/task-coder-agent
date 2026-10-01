import MarkdownLib from "react-markdown";
import remarkGfm from "remark-gfm";

/**
 * Renders LLM prose as CommonMark + GFM (tables, strikethrough, task lists).
 * Raw HTML is NOT rendered (react-markdown escapes it by default), and URLs
 * are passed through the default sanitizer (http/https/mailto/tel only), so
 * model output cannot inject markup or javascript: links.
 */
export function Markdown({ text }: { text: string }) {
  if (!text || !text.trim()) return null;
  return (
    <div className="markdown">
      <MarkdownLib
        remarkPlugins={[remarkGfm]}
        components={{
          a: ({ node, ...props }) => (
            <a {...props} target="_blank" rel="noopener noreferrer" />
          ),
        }}
      >
        {text}
      </MarkdownLib>
    </div>
  );
}
