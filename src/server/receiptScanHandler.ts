import fs from 'node:fs/promises';
import path from 'node:path';
import type { NextApiRequest, NextApiResponse } from 'next';
import { getServerSession } from 'next-auth/next';
import sharp from 'sharp';

import {
  type ReceiptScanProvider,
  getReceiptScanProvider,
} from '~/server/api/services/receiptScanService';
import { authOptions } from '~/server/auth';
import { fileExists } from '~/utils/file';

const UPLOAD_DIR = path.join(process.cwd(), 'uploads');

/* Shared by both scan endpoints: method, auth, ownership and path checks, then hands the
   image to `scan`. `errorMessage` is used for both the log and the 500 body. */
export async function handleReceiptScan(
  req: NextApiRequest,
  res: NextApiResponse,
  scan: (provider: ReceiptScanProvider, imageBase64: string, mimeType: string) => Promise<unknown>,
  errorMessage: string,
) {
  if ('POST' !== req.method) {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  /* Auth before provider resolution: the other order lets an anonymous caller tell a
     configured install (401) from an unconfigured one (501), and trigger the Ollama probe. */
  const session = await getServerSession(req, res, authOptions);
  if (!session?.user) {
    return res.status(401).json({ error: 'Unauthorized' });
  }

  const provider = await getReceiptScanProvider();
  if (!provider) {
    return res.status(501).json({ error: 'Receipt scanning is not configured' });
  }

  /* Next leaves the body undefined when it is empty or the Content-Type is unparsed. */
  const { fileKey } = (req.body ?? {}) as { fileKey?: unknown };
  if ('string' !== typeof fileKey || '' === fileKey) {
    return res.status(400).json({ error: 'fileKey is required' });
  }

  const userId = String(session.user.id);
  if (!fileKey.startsWith(`${userId}/`)) {
    return res.status(403).json({ error: 'Forbidden' });
  }

  const filePath = path.join(UPLOAD_DIR, fileKey);
  const resolvedPath = path.resolve(filePath);

  /* Compared with path.relative, not startsWith: a prefix test has no separator boundary, so
     a sibling `<root>/uploads-backup` passes it and traversal segments can reach there. */
  const relativeToUploads = path.relative(path.resolve(UPLOAD_DIR), resolvedPath);
  if (
    !relativeToUploads ||
    relativeToUploads.startsWith('..') ||
    path.isAbsolute(relativeToUploads)
  ) {
    return res.status(403).json({ error: 'Forbidden' });
  }

  if (!(await fileExists(filePath))) {
    return res.status(404).json({ error: 'File not found' });
  }

  try {
    const fileBuffer = await fs.readFile(filePath);
    // Uploads are stored as WebP, which llama.cpp-based servers often cannot decode.
    const jpegBuffer = await sharp(fileBuffer).jpeg().toBuffer();
    const imageBase64 = jpegBuffer.toString('base64');
    const mimeType = 'image/jpeg';

    const result = await scan(provider, imageBase64, mimeType);
    return res.status(200).json(result);
  } catch (error) {
    console.error(`${errorMessage}:`, error);
    return res.status(500).json({ error: errorMessage });
  }
}
