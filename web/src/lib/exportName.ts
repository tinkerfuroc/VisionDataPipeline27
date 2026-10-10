/** Folder id from a dataset name; must match server/vdp/export/naming.py `slug`. Empty = a timestamp id is used. */
export function slug(name: string): string {
  let s = name.trim().toLowerCase().replace(/[^a-z0-9._-]+/g, "-").replace(/-{2,}/g, "-");
  s = s.replace(/^[-._]+/, "");
  return s.slice(0, 48).replace(/[-._]+$/, "");
}

export const exportLabel = (e: { id: string; name: string | null }): string => e.name ?? e.id;
