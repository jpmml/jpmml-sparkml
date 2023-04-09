/*
 * Copyright (c) 2017 Villu Ruusmann
 *
 * This file is part of JPMML-SparkML
 *
 * JPMML-SparkML is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * JPMML-SparkML is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with JPMML-SparkML.  If not, see <http://www.gnu.org/licenses/>.
 */
package org.jpmml.sparkml;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.file.FileVisitResult;
import java.nio.file.FileVisitor;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.SimpleFileVisitor;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.Enumeration;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;
import java.util.zip.ZipOutputStream;

import com.google.common.io.ByteStreams;

public class ZipUtil {

	private ZipUtil(){
	}

	static
	public void compress(File dir, File file) throws Exception {
		Path dirPath = Paths.get(dir.getAbsolutePath());

		try(OutputStream os = new FileOutputStream(file)){
			ZipOutputStream zos = new ZipOutputStream(os);

			FileVisitor<Path> dirFileVisitor = new SimpleFileVisitor<Path>(){

				@Override
				public FileVisitResult visitFile(Path path, BasicFileAttributes mainAtts) throws IOException {
					File dirFile = path.toFile();

					Path relativePath = dirPath.relativize(path);

					ZipEntry entry = new ZipEntry(relativePath.toString());
					entry.setSize(dirFile.length());
					entry.setTime(dirFile.lastModified());

					zos.putNextEntry(entry);

					try(InputStream is = new FileInputStream(dirFile)){
						ByteStreams.copy(is, zos);
					}

					zos.closeEntry();

					return FileVisitResult.CONTINUE;
				}
			};

			Files.walkFileTree(dirPath, dirFileVisitor);

			zos.finish();
		}
	}

	static
	public void uncompress(File file, File dir) throws IOException {

		try(ZipFile zipFile = new ZipFile(file)){
			uncompress(zipFile, dir);
		}
	}

	static
	public void uncompress(ZipFile zipFile, File dir) throws IOException {

		for(Enumeration<? extends ZipEntry> entries = zipFile.entries(); entries.hasMoreElements(); ){
			ZipEntry entry = entries.nextElement();

			if(entry.isDirectory()){
				continue;
			}

			try(InputStream is = zipFile.getInputStream(entry)){
				File file = new File(dir, entry.getName());

				File parentDir = file.getParentFile();
				if(!parentDir.exists()){

					if(!parentDir.mkdirs()){
						throw new IOException(parentDir.getAbsolutePath());
					}
				}

				try(OutputStream os = new FileOutputStream(file)){
					ByteStreams.copy(is, os);
				}
			}
		}
	}
}